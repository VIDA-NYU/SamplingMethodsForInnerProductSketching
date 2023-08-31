import numpy as np
from .abstract_class import InnerProdSketcher, InnerProdSketch, hash_kwise
from numba import njit

#
# Priory Sampling Sketch
#
class PSSketch(InnerProdSketch):
    def __init__(self, sk_hashes: np.ndarray, sk_values: np.ndarray, tau: float, norm: int) -> None:
        self.sk_hashes: np.ndarray = sk_hashes
        self.sk_values: np.ndarray = sk_values
        self.tau: float = tau
        self.norm: int = norm
        
    @staticmethod
    @njit(parallel=False)
    def inner_product_numba(sk_hashesA, sk_valuesA, normA, tauA, sk_hashesB, sk_valuesB, normB, tauB):
        i = 0
        j = 0
        ip_est = 0
        cnt = 0
        while i < len(sk_hashesA) and j < len(sk_hashesB):
            ha, va = sk_hashesA[i], sk_valuesA[i]
            hb, vb = sk_hashesB[j], sk_valuesB[j]
            if ha == hb:
                denominator = min(1, ((va ** 2) ** (normA / 2)) * tauA, ((vb ** 2) ** (normB / 2)) * tauB)
                ip_est += (va * vb) / denominator
                cnt += 1
            if ha <= hb:
                i += 1
            else:
                j += 1
        # return ip_est
        return (ip_est, cnt)
    
    def inner_product(self, other: 'PSSketch') -> float:
        return self.inner_product_numba(self.sk_hashes, self.sk_values, self.norm, self.tau, other.sk_hashes, other.sk_values, other.norm, other.tau)

        # ip_est = 0
        # cnt = 0
        # for h in self.sk_hashes:
        #     if h in other.sk_hashes:
        #         ia = np.where(self.sk_hashes == h)[0][0]
        #         ib = np.where(other.sk_hashes == h)[0][0]
        #         va = self.sk_values[ia]
        #         vb = other.sk_values[ib]
        #         denominator = min(1, ((va**2)**(self.norm/2))*self.tau, ((vb**2)**(self.norm/2))*other.tau)
        #         ip_est += (va * vb)/denominator
        #         cnt+=1
        # print(f"cnt: {cnt}")
        # return (ip_est, cnt)


class PS(InnerProdSketcher):
    def __init__(self, sketch_size: int, seed: int, norm: int) -> None:
        self.sketch_size: int = sketch_size
        self.seed: int = seed
        self.norm: int = norm
        
    def sketch(self, vector: np.ndarray) -> PSSketch:
        hashes, values = hash_kwise(vector, self.seed)
        ranks = hashes/((values**2)**(self.norm/2))
        try:
            tau = np.partition(ranks, self.sketch_size-1)[self.sketch_size-1]
        except:
            tau = ranks[-1] # if the sketch size is larger than the number of non-zero elements
        indices_under_tau = ranks <= tau
        sk_hashes = hashes[indices_under_tau]
        sk_values = values[indices_under_tau]

        # this sort is for optimizing the inner product computation
        k_min = np.argsort(sk_hashes)[:self.sketch_size]
        sk_hashes = sk_hashes[k_min]
        sk_values = sk_values[k_min]
        return PSSketch(sk_hashes, sk_values, tau, self.norm)