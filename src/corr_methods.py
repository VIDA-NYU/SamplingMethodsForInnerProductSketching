import numpy as np
from .abstract_class import InnerProdSketcher, InnerProdSketch, hash_kwise
import heapq

def compute_adaptive_threshold_corr(vector, target_size, delta=0.05):
    vector_l0 = np.linalg.norm(vector, ord=0)
    vector_l2 = np.linalg.norm(vector, ord=2)
    vector_l4 = np.linalg.norm(vector, ord=4)
    C = np.argsort(vector)[-target_size:]
    C = set(C.tolist())
    S = set()
    T = target_size
    sum_min = sum([min(1, T * max((val!=0)/vector_l0, (val/vector_l2)**2, (val/vector_l4)**4)) for val in vector])
    if sum_min <= target_size:
        return T
    print(f"===\nbefore while loop\nsum_min: {sum_min}\ntarget_size: {target_size}\nT: {T}\ndelta: {delta}")
    while sum_min > target_size:
        T -= 10
        sum_min = sum(
            [min(1, T * max((val!=0)/vector_l0, (val/vector_l2) ** 2, (val/vector_l4) ** 4)) for val in vector])
        print(f"---\n1st while loop\nsum_min: {sum_min}\ntarget_size: {target_size}\nT: {T}\ndelta: {delta}")
        
    while sum_min <= target_size*(1-delta):
        O = set([i for i in C if T *  max((vector[i]!=0)/vector_l0, (vector[i]/vector_l2)**2, (vector[i]/vector_l4)**4) >= 1])
        S |= O
        C -= O
        T = (target_size - len(S)) / sum([max((vector[i]!=0)/vector_l0, (vector[i]/vector_l2)**2, (vector[i]/vector_l4)**4) for i in range(len(vector)) if i not in S])
        sum_min = sum([min(1, T *  max((val!=0)/vector_l0, (val/vector_l2)**2, (val/vector_l4)**4)) for val in vector])
        print(f"---\n2nd while loop\nsum_min: {sum_min}\ntarget_size: {target_size}\nT: {T}\ndelta: {delta}")
    return T

#
# Threshold Sampling Sketch for Correlation
#
class TSCorrSketch(InnerProdSketch):
    def __init__(self, sk_indices: np.ndarray, sk_values: np.ndarray, 
                 threshold_unweighted: float, threshold_weighted: float,
                 vector_l2: float, vector_l4: float) -> None:
        self.sk_indices: np.ndarray = sk_indices
        self.sk_values: np.ndarray = sk_values
        self.sample_sk_values: np.ndarray = None
        self.threshold_unweighted: float = threshold_unweighted
        self.threshold_weighted: float = threshold_weighted
        self.vector_l2: float = vector_l2
        self.vector_l4: float = vector_l4
        
    def inner_product(self, other: 'TSCorrSketch') -> float:
        ip_est = 0
        for iia, ia in enumerate(self.sk_indices):
            if ia in other.sk_indices:
                iib = np.where(other.sk_indices == ia)[0][0]
                va, vb = self.sk_values[iia], other.sk_values[iib]
                s_va, s_vb = self.sample_sk_values[iia], other.sample_sk_values[iib]
                va2, vb2 = (s_va/self.vector_l2)**2, (s_vb/other.vector_l2)**2
                va4, vb4 = (s_va/self.vector_l4)**4, (s_vb/other.vector_l4)**4
                denominator = min(1, 
                                  max(self.threshold_weighted * va2, self.threshold_weighted * va4, self.threshold_unweighted),
                                  max(other.threshold_weighted * vb2, other.threshold_weighted * vb4, other.threshold_unweighted))
                ip_est += (va * vb)/denominator
        return ip_est


class TSCorr(InnerProdSketcher):
    def __init__(self, sketch_size: int, seed: int) -> None:
        self.sketch_size: int = sketch_size
        self.seed: int = seed

    def sketch(self, vector: np.ndarray) -> TSCorrSketch:
        hashes, values = hash_kwise(vector, self.seed)
        vector_nonzeroIndex = np.nonzero(vector)[0]
        nonzero_size = vector_nonzeroIndex.shape[0]
        vector_l0, vector_l2, vector_l4 = np.linalg.norm(vector, ord=0), np.linalg.norm(vector, ord=2), np.linalg.norm(vector, ord=4)
        # vector_norm0, vector_norm2, vector_norm4 = (vector!=0)/vector_l0, (vector/vector_l2)**2, (vector/vector_l4)**4

        # m = compute_adaptive_threshold_corr(vector, self.sketch_size)
        # hashes_indices = (hashes <= m * 1/nonzero_size) | (hashes <= m * vector_norm2[vector_nonzeroIndex]) | (
        #                 hashes <= m * vector_norm4[vector_nonzeroIndex])
        m = compute_adaptive_threshold_corr(vector, self.sketch_size)
        vector_nz = vector[vector_nonzeroIndex]
        hashes_indices = []
        for hi, hash in enumerate(hashes):
            ai = vector_nz[hi]
            ti_1a = m * 1/vector_l0
            ti_a = m * (ai/vector_l2)**2
            ti_a2 = m * (ai/vector_l4)**4
            Ti = max(ti_1a, ti_a, ti_a2)
            if hash <= Ti:
                hashes_indices.append(hi)
        sk_indices = vector_nonzeroIndex[hashes_indices]
        sk_values = values[hashes_indices]
        return TSCorrSketch(sk_indices, sk_values, m/nonzero_size, m, vector_l2, vector_l4)


# 
# Priority Sampling Sketch for Correlation
# 
class PSCorrSketch(InnerProdSketch):
    def __init__(self, sk_indices: np.ndarray, sk_values: np.ndarray, 
                 threshold_unweighted: float, threshold_weighted: float, threshold_weighted2: float,
                 vector_l2: float, vector_l4: float) -> None:
        self.sk_indices: np.ndarray = sk_indices
        self.sk_values: np.ndarray = sk_values
        self.sample_sk_values: np.ndarray = None
        self.threshold_unweighted: float = threshold_unweighted
        self.threshold_weighted: float = threshold_weighted
        self.threshold_weighted2: float = threshold_weighted2
        self.vector_l2: float = vector_l2
        self.vector_l4: float = vector_l4

    def inner_product(self, other: 'PSCorrSketch') -> float:
        ip_est = 0
        for iia, ia in enumerate(self.sk_indices):
            if ia in other.sk_indices:
                iib = np.where(other.sk_indices == ia)[0][0]
                va, vb = self.sk_values[iia], other.sk_values[iib]
                s_va, s_vb = self.sample_sk_values[iia], other.sample_sk_values[iib]
                va2, vb2 = (s_va/self.vector_l2)**2, (s_vb/other.vector_l2)**2
                va4, vb4 = (s_va/self.vector_l4)**4, (s_vb/other.vector_l4)**4
                denominator = min(1, 
                                  max(self.threshold_weighted * va2, self.threshold_weighted2 * va4, self.threshold_unweighted),
                                  max(other.threshold_weighted * vb2, other.threshold_weighted2 * vb4, other.threshold_unweighted))
                ip_est += (va * vb)/denominator
        return ip_est
    

class PSCorr(InnerProdSketcher):
    def __init__(self, sketch_size: int, seed: int) -> None:
        self.sketch_size: int = sketch_size
        self.seed: int = seed

    def sketch(self, vector: np.ndarray) -> PSCorrSketch:
        hashes, values = hash_kwise(vector, self.seed)
        vector_nonzeroIndex = np.nonzero(vector)[0]
        vector_l2, vector_l4 = np.linalg.norm(vector, ord=2), np.linalg.norm(vector, ord=4)
        vector_norm2, vector_norm4 = (vector/vector_l2)**2, (vector/vector_l4)**4

        priority_norm0 = hashes
        priority_norm1 = hashes / vector_norm2[vector_nonzeroIndex]
        priority_norm2 = hashes / vector_norm4[vector_nonzeroIndex]

        k_min = 0
        k_max = self.sketch_size
        num_selected = 0
        while k_min < k_max:
            k_mid = (k_min + k_max) // 2
            threshold_unweighted = heapq.nsmallest(k_mid+1, priority_norm0)[-1]
            threshold_weighted = heapq.nsmallest(k_mid+1, priority_norm1)[-1]
            threshold_weighted2 = heapq.nsmallest(k_mid+1, priority_norm2)[-1]
            hashes_index = (hashes < threshold_unweighted) | (hashes < threshold_weighted * vector_norm2[vector_nonzeroIndex]) | (
                        hashes < threshold_weighted2 * vector_norm4[vector_nonzeroIndex])
            num_selected = np.sum(hashes_index)
            if num_selected == self.sketch_size:
                break
            elif num_selected > self.sketch_size:
                k_max = k_mid
            else:
                k_min = k_mid + 1

        sk_indices = vector_nonzeroIndex[hashes_index]
        sk_values = values[hashes_index]
        return PSCorrSketch(sk_indices, sk_values, threshold_unweighted, threshold_weighted, threshold_weighted2, vector_l2, vector_l4)
    

#
# Threshold Sampling Sketch for Correlation
#
class TS012CorrSketch(InnerProdSketch):
    def __init__(self, sk_indices: np.ndarray, sk_values: np.ndarray, 
                 threshold_unweighted: float, threshold_weighted: float,
                 vector_l1: float, vector_l2: float) -> None:
        self.sk_indices: np.ndarray = sk_indices
        self.sk_values: np.ndarray = sk_values
        self.sample_sk_values: np.ndarray = None
        self.threshold_unweighted: float = threshold_unweighted
        self.threshold_weighted: float = threshold_weighted
        self.vector_l1: float = vector_l1
        self.vector_l2: float = vector_l2
        
    def inner_product(self, other: 'TS012CorrSketch') -> float:
        ip_est = 0
        for iia, ia in enumerate(self.sk_indices):
            if ia in other.sk_indices:
                iib = np.where(other.sk_indices == ia)[0][0]
                va, vb = self.sk_values[iia], other.sk_values[iib]
                s_va, s_vb = self.sample_sk_values[iia], other.sample_sk_values[iib]
                va1, vb1 = ((s_va/self.vector_l1)**2)**0.5, ((s_vb/other.vector_l1)**2)**0.5
                va2, vb2 = (s_va/self.vector_l2)**2, (s_vb/other.vector_l2)**2
                denominator = min(1, 
                                  max(self.threshold_weighted * va1, self.threshold_weighted * va2, self.threshold_unweighted),
                                  max(other.threshold_weighted * vb1, other.threshold_weighted * vb2, other.threshold_unweighted))
                ip_est += (va * vb)/denominator
        return ip_est


class TS012Corr(InnerProdSketcher):
    def __init__(self, sketch_size: int, seed: int) -> None:
        self.sketch_size: int = sketch_size
        self.seed: int = seed

    def sketch(self, vector: np.ndarray) -> TS012CorrSketch:
        hashes, values = hash_kwise(vector, self.seed)
        vector_nonzeroIndex = np.nonzero(vector)[0]
        nonzero_size = vector_nonzeroIndex.shape[0]
        vector_l1, vector_l2 = np.linalg.norm(vector, ord=1), np.linalg.norm(vector, ord=2)
        vector_norm1, vector_norm2 = ((vector/vector_l1)**2)**0.5, (vector/vector_l2)**2

        k_min = 0
        k_max = self.sketch_size
        num_selected = 0
        while k_min < k_max:
            k_mid = (k_min + k_max) // 2
            threshold_unweighted = k_mid/nonzero_size
            threshold_weighted = k_mid
            hashes_indices = (hashes <= threshold_unweighted) | (hashes <= threshold_weighted * vector_norm1[vector_nonzeroIndex]) | (
                        hashes <= threshold_weighted * vector_norm2[vector_nonzeroIndex])
            num_selected = np.sum(hashes_indices)
            if num_selected == self.sketch_size:
                break
            elif num_selected > self.sketch_size:
                k_max = k_mid
            else:
                k_min = k_mid + 1
        
        sk_indices = vector_nonzeroIndex[hashes_indices]
        sk_values = values[hashes_indices]
        return TS012CorrSketch(sk_indices, sk_values, threshold_unweighted, threshold_weighted, vector_l1, vector_l2)


# 
# Priority Sampling Sketch for Correlation
# 
class PS012CorrSketch(InnerProdSketch):
    def __init__(self, sk_indices: np.ndarray, sk_values: np.ndarray, 
                 threshold_unweighted: float, threshold_weighted: float, threshold_weighted2: float,
                 vector_l1: float, vector_l2: float) -> None:
        self.sk_indices: np.ndarray = sk_indices
        self.sk_values: np.ndarray = sk_values
        self.sample_sk_values: np.ndarray = None
        self.threshold_unweighted: float = threshold_unweighted
        self.threshold_weighted: float = threshold_weighted
        self.threshold_weighted2: float = threshold_weighted2
        self.vector_l1: float = vector_l1
        self.vector_l2: float = vector_l2

    def inner_product(self, other: 'PS012CorrSketch') -> float:
        ip_est = 0
        for iia, ia in enumerate(self.sk_indices):
            if ia in other.sk_indices:
                iib = np.where(other.sk_indices == ia)[0][0]
                va, vb = self.sk_values[iia], other.sk_values[iib]
                s_va, s_vb = self.sample_sk_values[iia], other.sample_sk_values[iib]
                va1, vb1 = ((s_va/self.vector_l1)**2)**0.5, ((s_vb/other.vector_l1)**2)**0.5
                va2, vb2 = (s_va/self.vector_l2)**2, (s_vb/other.vector_l2)**2
                denominator = min(1, 
                                  max(self.threshold_weighted * va1, self.threshold_weighted2 * va2, self.threshold_unweighted),
                                  max(other.threshold_weighted * vb1, other.threshold_weighted2 * vb2, other.threshold_unweighted))
                ip_est += (va * vb)/denominator
        return ip_est
    

class PS012Corr(InnerProdSketcher):
    def __init__(self, sketch_size: int, seed: int) -> None:
        self.sketch_size: int = sketch_size
        self.seed: int = seed

    def sketch(self, vector: np.ndarray) -> PS012CorrSketch:
        hashes, values = hash_kwise(vector, self.seed)
        vector_nonzeroIndex = np.nonzero(vector)[0]
        vector_l1, vector_l2 = np.linalg.norm(vector, ord=1), np.linalg.norm(vector, ord=2)
        vector_norm1, vector_norm2 = ((vector/vector_l1)**2)**0.5, (vector/vector_l2)**2

        priority_norm0 = hashes
        priority_norm1 = hashes / vector_norm1[vector_nonzeroIndex]
        priority_norm2 = hashes / vector_norm2[vector_nonzeroIndex]

        k_min = 0
        k_max = self.sketch_size
        num_selected = 0
        while k_min < k_max:
            k_mid = (k_min + k_max) // 2
            threshold_unweighted = heapq.nsmallest(k_mid+1, priority_norm0)[-1]
            threshold_weighted = heapq.nsmallest(k_mid+1, priority_norm1)[-1]
            threshold_weighted2 = heapq.nsmallest(k_mid+1, priority_norm2)[-1]
            hashes_index = (hashes < threshold_unweighted) | (hashes < threshold_weighted * vector_norm1[vector_nonzeroIndex]) | (
                        hashes < threshold_weighted2 * vector_norm2[vector_nonzeroIndex])
            num_selected = np.sum(hashes_index)
            if num_selected == self.sketch_size:
                break
            elif num_selected > self.sketch_size:
                k_max = k_mid
            else:
                k_min = k_mid + 1

        sk_indices = vector_nonzeroIndex[hashes_index]
        sk_values = values[hashes_index]
        return PS012CorrSketch(sk_indices, sk_values, threshold_unweighted, threshold_weighted, threshold_weighted2, vector_l1, vector_l2)