import numpy as np
import scipy.sparse as sparse
from .abstract_class import InnerProdSketcher, InnerProdSketch

#
# Johnson-Lindenstrauss Sketch
#
class JLSketch(InnerProdSketch):
    def __init__(self, sk_values: np.ndarray) -> None:
        self.sk_values: np.ndarray = sk_values

    def inner_product(self, other: 'JLSketch') -> float:
        return self.sk_values.dot(other.sk_values)


class JL(InnerProdSketcher):
    def __init__(self, sketch_size: int, seed: int, prime: int = 2147483587, k_wise: int=4) -> None:
        # Initialize the JL object with given parameters
        self.sketch_size: int = sketch_size  # Number of rows in the sketch matrix
        self.pi_rows: int = sketch_size  # Number of rows in the sketch matrix
        self.seed: int = seed  # Random seed for reproducibility
        self.prime: int = prime  # A prime number used in hash function
        self.k_wise: int = k_wise  # The k-wise independence of the hash function

    def sketch(self, vector: np.ndarray) -> JLSketch:
        np.random.seed(self.seed)
        # Generate random hash parameters for the sketch
        hash_parameters = np.random.randint(1, self.prime, (self.pi_rows, self.k_wise))

        # Find the indices of nonzero elements in the vector
        nonzero_index = sparse.find(vector != 0)[1]

        # Compute the sketch matrix using the Johnson-Lindenstrauss transform
        matrix_pi = 0
        for exp in range(self.k_wise):
            matrix_pi += np.dot(np.transpose(np.array([nonzero_index]) ** exp), np.array([np.transpose(hash_parameters[:, exp])]))
        matrix_pi = np.mod(np.mod(matrix_pi, self.prime), 2) * 2 - 1
        matrix_pi = matrix_pi * (1 / np.sqrt(self.pi_rows))

        # Compute the sketch values by taking dot product with the vector
        sk_values = matrix_pi.T.dot(vector[nonzero_index])

        # Return the JL sketch as a JLSketch object
        return JLSketch(sk_values)


#
# Count Sketch
#
class CSSketch(InnerProdSketch):
    def __init__(self, sk_values: np.ndarray) -> None:
        self.sk_values: np.ndarray = sk_values

    def inner_product(self, other: 'CSSketch') -> float:
        return np.median(np.sum(self.sk_values * other.sk_values, axis=1))

class CS(InnerProdSketcher):
    def __init__(self, sketch_size: int, seed: int, prime: int = 2147483587, t: int=3) -> None:
        self.sketch_size: int = sketch_size
        self.seed: int = seed
        self.prime: int = prime
        self.t: int = t # determines the number of hash functions

    def sketch(self, vector: np.ndarray) -> CSSketch:
        vec_ind = [i for i in range(len(vector))]
        np.random.seed(self.seed)
        num_of_hashes = 2*self.t-1
        seeds = np.random.randint(0, 10000 + 1, size=2*num_of_hashes)
        hs, gs = [], []
        for i in range(num_of_hashes):
            np.random.seed(seeds[i])
            h_para = np.random.randint(1, self.prime, (1, 2))
            h = np.mod(np.mod(h_para[:,0]*vec_ind+h_para[:,1], self.prime), self.sketch_size)
            hs.append(h)

            np.random.seed(seeds[i+num_of_hashes])
            g_para = np.random.randint(1, self.prime, (1, 2))
            g = np.mod(np.mod(g_para[:,0]*vec_ind+g_para[:,1], self.prime), 2)*2-1
            gs.append(g)

        sk_values = np.zeros((num_of_hashes, self.sketch_size))
        for row_num, (h, g) in enumerate(zip(hs, gs)):
            for vi, hi, gi in zip(vector, h, g):
                sk_values[row_num, hi] += gi * vi

        return CSSketch(sk_values)

