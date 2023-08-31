import numpy as np
import scipy.stats as stats
import pandas as pd
import uuid
from scipy.stats import pearsonr


def create_vector_pair(nonZero_size=2000, mean=0, std=1, random_seed=9):
    """
    Create a pair of random vectors.

    Args:
        nonZero_size (int, optional): Number of non-zero elements in the vectors. Defaults to 2000.
        mean (float, optional): Mean value for the distribution. Defaults to 0.
        std (float, optional): Standard deviation for the distribution. Defaults to 1.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 9.

    Returns:
        tuple: A tuple containing two numpy arrays representing the created vectors.
            - vecA_nonZero (numpy.ndarray): Vector A with non-zero elements.
            - vecB_nonZero (numpy.ndarray): Vector B with non-zero elements.
    """
    np.random.seed(random_seed)

    loc, scale = mean-std, std*2
    
    # Generate non-zero elements for vectors A and B from a uniform distribution [loc, loc+scale] or [-1,1]
    vecA_nonZero = stats.uniform.rvs(loc=loc, scale=scale, size=nonZero_size)
    vecB_nonZero = stats.uniform.rvs(loc=loc, scale=scale, size=nonZero_size)
    
    # Randomly shuffle the non-zero elements in both vectors using the same indices
    randIndex = np.random.choice(nonZero_size, size=nonZero_size, replace=False)
    vecA_nonZero = vecA_nonZero[randIndex]
    vecB_nonZero = vecB_nonZero[randIndex]
    
    return vecA_nonZero, vecB_nonZero


def create_vector_pair_with_outlier(nonZero_size=2000, mean=0, std=1, random_seed=9, 
                                    mean_outlier=25, std_outlier=5, outlier_fraction=0.01):
    vecA_nonZero, vecB_nonZero = create_vector_pair(nonZero_size, mean, std, random_seed)
    
    low, upp = mean_outlier-std_outlier, mean_outlier+std_outlier
    X2 = stats.truncnorm((low-mean_outlier)/std_outlier, (upp-mean_outlier)/std_outlier, loc=mean_outlier, scale=std_outlier)
    outlier_size = round(nonZero_size*outlier_fraction)
    vecA_outlier = X2.rvs(outlier_size)
    vecB_outlier = X2.rvs(outlier_size)
    vecA_outlier = np.array([round(i) for i in vecA_outlier])
    vecB_outlier = np.array([round(i) for i in vecB_outlier])
    outlier_index = np.random.choice(nonZero_size, size=outlier_size, replace=False)
    np.put(vecA_nonZero, outlier_index, vecA_outlier)
    np.put(vecB_nonZero, outlier_index, vecB_outlier)
    return vecA_nonZero, vecB_nonZero

def create_sparse_vector_pair(vecA_nonZero, vecB_nonZero, overlap_ratio, 
                              sparse_vec_size=10000, nonZero_size=2000):
    nonZero_index = np.random.choice(sparse_vec_size, size=nonZero_size*2, replace=False)
    nonZero_indexA = nonZero_index[:nonZero_size]
    nonZero_indexB = nonZero_index[nonZero_size:]
    intersec_size = round(nonZero_size*overlap_ratio)
    intersec_index = nonZero_indexA[:intersec_size]
    nonZero_indexB = np.hstack((intersec_index, 
                                nonZero_indexB[intersec_size:]))
    vecA = np.zeros(sparse_vec_size)
    vecB = np.zeros(sparse_vec_size)
    np.put(vecA, nonZero_indexA, vecA_nonZero)
    np.put(vecB, nonZero_indexB, vecB_nonZero)
    return vecA, vecB


def create_correlated_dataset(initial_dataset, new_dataset, r, nrows):
    K = [uuid.uuid1().hex for i in range(nrows)]
    df = pd.DataFrame({'K': K,'X': new_dataset, 'Y': initial_dataset})
    import statsmodels.formula.api as smf
    linmodel = smf.ols(formula="X ~ Y", data=df).fit()
    perpendicular = linmodel.resid
    correlated_dataset = r*np.std(perpendicular)*initial_dataset + perpendicular*np.std(initial_dataset)*np.sqrt(1-r**2)
    #print('corr', pearsonr(initial_dataset, correlated_dataset))
    df_corr = pd.DataFrame({'K': K, 'Corr': correlated_dataset})
    return df_corr


def get_vecAvecB(overlap_ratio, outlier_fraction, mode, corr_r, nonZero_size=2000):
    sparse_vec_size = nonZero_size*5
    vecA_nonZero, vecB_nonZero = create_vector_pair_with_outlier(nonZero_size=nonZero_size, mean_outlier=5, std_outlier=1, outlier_fraction=outlier_fraction)
    if mode == 'corr':
        print(f"correlation before: {pearsonr(vecA_nonZero, vecB_nonZero)}")
        vecC = create_correlated_dataset(vecA_nonZero, vecB_nonZero, corr_r, vecA_nonZero.shape[0])
        vecB_nonZero = vecC['Corr'].to_numpy()
        print(f"correlation after: {pearsonr(vecA_nonZero, vecB_nonZero)}")
    vecA, vecB = create_sparse_vector_pair(vecA_nonZero, vecB_nonZero, overlap_ratio, sparse_vec_size=sparse_vec_size, nonZero_size=nonZero_size)
    if mode == 'join_size':
        vecA = np.array([1 if i!=0 else 0 for i in vecA])
        vecB = np.array([1 if i!=0 else 0 for i in vecB])
    return vecA,vecB