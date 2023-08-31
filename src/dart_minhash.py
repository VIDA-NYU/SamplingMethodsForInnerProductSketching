import numpy as np
import random
import math
from .abstract_class import InnerProdSketcher, InnerProdSketch

class DartMHSketch(InnerProdSketch):
    def __init__(self, sk_hashes: np.ndarray, sk_indices: np.ndarray, sk_values: np.ndarray, vec_l2: float) -> None:
        self.sk_hashes: np.ndarray = sk_hashes
        self.sk_indices: np.ndarray = sk_indices
        self.sk_values: np.ndarray = sk_values
        self.vec_l2: float = vec_l2

    def inner_product(self, other: 'DartMHSketch') -> float:
        m = len(self.sk_hashes)  # sample size
        mean_min = np.mean([min(hA, hB) for hA, hB in zip(self.sk_hashes, other.sk_hashes)])
        M_est = (1 / mean_min - 1)
        sum_m = sum([(va * vb) / min(va ** 2, vb ** 2) for ha, hb, va, vb in zip(self.sk_hashes, other.sk_hashes, self.sk_values, other.sk_values) if ha == hb])
        wmh_est = self.vec_l2 * other.vec_l2 * (M_est / m) * sum_m
        return wmh_est
    
class DartMH(InnerProdSketcher):
    def __init__(self, sketch_size: int, seed: int) -> None:
        self.sketch_size: int = sketch_size
        self.seed: int = seed
        self.phi_range = (1, 50, 1)

    def hash(self, vec, vec_l1, phi, t, seed, sample_size):
        D = []
        vec_nonzeroIndex = np.nonzero(vec)[0]
        
        for i in vec_nonzeroIndex:
            xi = abs(vec[i])
            
            for v in range(math.ceil(math.log(1 + t * xi, 2))):
                for rho in range(math.ceil(math.log(1 + phi / vec_l1, 2))):
                    W, R = (2 ** v - 1), (2 ** rho - 1)
                    delta_v, delta_rho = (2 ** v) / (t * (2 ** rho)), (2 ** rho) / (2 ** v)
                    
                    for w in range(2 ** rho):
                        if xi < (W + w * delta_v):
                            break
                        for r in range(2 ** v):
                            if (phi / vec_l1) < (R + r * delta_rho):
                                break
                            j = 0
                            seed_num = 99998111 * phi + 9998867 * v + 998629 * rho + 7907 * w + 107 * r + seed + i * 999997769
                            random.seed(seed_num % (2147483647))
                            X = np.random.poisson(1)
                            
                            while j < X:
                                V = random.uniform(0.0, 1.0)
                                U = random.uniform(0.0, 1.0)
                                color = random.randint(0, sample_size - 1)
                                weight = W + (w + V) * delta_v
                                rank = R + (r + U) * delta_rho
                                index = (i, v, rho, w, r, j)
                                
                                if weight <= xi and rank <= (phi / vec_l1):
                                    D.append((index, rank / (phi / vec_l1), color))
                                j += 1
        return D

    def sketch(self, vector: np.ndarray) -> DartMHSketch:
        vec_l2 = np.linalg.norm(vector, ord=2)
        vec_norm = vector / vec_l2
        vec_norm2 = vec_norm ** 2
        vec = vec_norm2
        
        t = math.ceil(self.sketch_size * math.log(self.sketch_size)) + self.sketch_size
        Darts = []
        Colors = []
        vec_l1 = np.linalg.norm(vec, ord=1)
        
        np.random.seed(self.seed)
        for phi in range(*self.phi_range):
            Q = self.hash(vec, vec_l1, phi, t, self.seed, self.sketch_size)
            colors = [q[2] for q in Q]
            Darts += Q
            Colors += colors
            if np.unique(Colors).shape[0] == self.sketch_size:
                break
        
        sk_hashes = [1.1] * self.sketch_size
        sk_indices = [0] * self.sketch_size
        sk_values = [0] * self.sketch_size
        for dart in Darts:
            index, rank, color = dart
            i = index[0]
            value = vec[i]
            
            if rank < sk_hashes[color]:
                sk_hashes[color] = rank
                sk_indices[color] = i
                sk_values[color] = value
        
        return DartMHSketch(sk_hashes, sk_indices, sk_values, vec_l2)