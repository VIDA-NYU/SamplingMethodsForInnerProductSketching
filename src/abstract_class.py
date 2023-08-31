from abc import ABC, abstractmethod
import numpy as np
from scipy import sparse

class InnerProdSketch(ABC):
    # All inner product sketches must implement this interface
    @abstractmethod
    def inner_product(self, other) -> float:
        pass


class InnerProdSketcher(ABC):
    @abstractmethod
    def sketch(self, a: np.array) -> InnerProdSketch:
        '''
        Each sketching algorithm must implement this function and return a
        InnerProductSketch object that implements the inner_product() function.
        '''
        pass
    
    
def compute_adaptive_threshold(vector, target_size, l_norm=2, delta=1e-9):
        vector_l = np.linalg.norm(vector, ord=l_norm)
        C = np.argsort(vector)[-target_size:]
        C = set(C.tolist())
        S = set()
        T = target_size
        sum_min = sum([min(1, T * (val / vector_l) ** l_norm) for val in vector])
        while sum_min <= target_size - delta:
            # print(f"sum_min: {sum_min}\ntarget_size: {target_size}\nT: {T}\ndelta: {delta}")
            O = set([i for i in C if T * (vector[i] / vector_l) ** l_norm >= 1])
            S |= O
            C -= O
            T = (target_size - len(S)) / sum([(vector[i] / vector_l) ** l_norm for i in range(len(vector)) if i not in S])
            sum_min = sum([min(1, T * (val / vector_l) ** l_norm) for val in vector])
        return T


def hash_kwise(vector, seed, dimension_num=1, k_wise=4, PRIME=2147483587):
    # if input is vector, treat index of values as keys
    keys = [i for i,v in enumerate(vector) if v != 0]
    values = [v for v in vector if v != 0]
    keys = np.array(keys, dtype=np.int32)
    values = np.array(values)
    return hash_kwise_kv(keys, values, seed, dimension_num, k_wise, PRIME)


def hash_kwise_kv(keys, values, seed, dimension_num=1, k_wise=4, PRIME=2147483587):
    np.random.seed(seed)
    hash_parameters = np.random.randint(1, PRIME, (dimension_num, k_wise))
    hash_kwise = 0
    for exp in range(k_wise):
        hash_kwise += np.dot(np.transpose(np.array([keys])**exp), np.array([np.transpose(hash_parameters[:, exp])]))
    hash_kwise = np.mod(hash_kwise, PRIME)/PRIME
    if dimension_num == 1:
        # Reshape the hash values as a 1D array
        hashes = hash_kwise.reshape(hash_kwise.shape[0],)
    else:
        # Find the minimum hash value for each column
        hashes = np.min(hash_kwise, axis=0)
        positions = np.argmin(hash_kwise, axis=0)
        values = values[positions]
    return hashes, values