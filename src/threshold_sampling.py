import numpy as np
from .abstract_class import InnerProdSketcher, InnerProdSketch, compute_adaptive_threshold, hash_kwise
from numba import njit

#
# Threshold Sampling Sketch
#
class TSSketch(InnerProdSketch):
    def __init__(self, sk_indices: np.ndarray, sk_values: np.ndarray, threshold: float, vector_norm: float, norm: int) -> None:
        self.sk_indices: np.ndarray = sk_indices
        self.sk_values: np.ndarray = sk_values
        self.threshold: float = threshold
        self.vector_norm: float = vector_norm
        self.norm: int = norm
        
    @staticmethod
    @njit(parallel=False)
    def inner_product_numba(sk_indicesA, sk_valuesA, normA, thresholdA, vector_normA, sk_indicesB, sk_valuesB, normB, thresholdB, vector_normB):
        i = 0
        j = 0
        ip_est = 0
        cnt = 0
        while i < len(sk_indicesA) and j < len(sk_indicesB):
            ia, va = sk_indicesA[i], sk_valuesA[i]
            ib, vb = sk_indicesB[j], sk_valuesB[j]
            if ia == ib:
                if normA == 0:
                        denominator = min(1, 
                                    thresholdA * (1 / vector_normA), 
                                    thresholdB * (1 / vector_normB))
                else:
                    denominator = min(1, 
                                    thresholdA * ((va / vector_normA) ** 2)**(normA/2), 
                                    thresholdB * ((vb / vector_normB) ** 2)**(normB/2))
                ip_est += va * vb / denominator
                cnt += 1
            if ia <= ib:
                i += 1
            else:
                j += 1
        # return ip_est
        return (ip_est, cnt)
    
    def inner_product(self, other: 'TSSketch') -> float:
        return self.inner_product_numba(self.sk_indices, self.sk_values, self.norm, self.threshold, self.vector_norm, other.sk_indices, other.sk_values, other.norm, other.threshold, other.vector_norm)
        
        # ip_est = 0
        # cnt = 0
        # for iia, ia in enumerate(self.sk_indices):
        #     if ia in other.sk_indices:
        #         iib = np.where(other.sk_indices == ia)[0][0]
        #         va = self.sk_values[iia]
        #         vb = other.sk_values[iib]
        #         if self.norm == 0:
        #             denominator = min(1, 
        #                           self.threshold * (1 / self.vector_norm), 
        #                           other.threshold * (1 / other.vector_norm))
        #         else:
        #             denominator = min(1, 
        #                             self.threshold * ((va / self.vector_norm) ** 2)**(self.norm/2), 
        #                             other.threshold * ((vb / other.vector_norm) ** 2)**(other.norm/2))
        #         ip_est += va * vb / denominator
        #         cnt+=1
        # print(f"cnt: {cnt}")
        # return (ip_est, cnt)


class TS(InnerProdSketcher):
    def __init__(self, sketch_size: int, seed: int, norm: int) -> None:
        self.sketch_size: int = sketch_size
        self.seed: int = seed
        self.norm: int = norm

    def sketch(self, vector: np.ndarray) -> TSSketch:
        vector_norm = np.linalg.norm(vector, ord=self.norm)
        vector_nonzeroIndex = np.nonzero(vector)[0]
        T = compute_adaptive_threshold(abs(vector), self.sketch_size, l_norm=self.norm)
        hashes, values = hash_kwise(vector, self.seed)
        if self.norm == 0:
            index_under_threshold = hashes <= T * (1/vector_norm)
        else:
            index_under_threshold = hashes <= T * ((vector[vector_nonzeroIndex]/vector_norm)**2)**(self.norm/2)
        sk_indices = vector_nonzeroIndex[index_under_threshold]
        sk_values = values[index_under_threshold]

        k_min = np.argsort(sk_indices)[:sk_indices.size]
        sk_indices = sk_indices[k_min]
        sk_values = sk_values[k_min]
        return TSSketch(sk_indices, sk_values, T, vector_norm, self.norm)