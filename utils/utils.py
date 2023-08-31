import numpy as np
from src import JL, CS, KMV, MH, WMH, TS, PS, TSCorr, PSCorr, DartMH
from src import TS012Corr, PS012Corr

def true_values(vecA, vecB):
    iA = np.array([1 if i!=0 else 0 for i in vecA])
    iB = np.array([1 if i!=0 else 0 for i in vecB])
    vecA2 = vecA**2
    vecB2 = vecB**2

    ip = vecA.dot(vecB)
    n = iA.dot(iB)
    sumA = vecA.dot(iB)
    sumB = vecB.dot(iA)
    sumA2 = vecA2.dot(iB)
    sumB2 = vecB2.dot(iA)
    
    corr = compute_correlation(ip, n, sumA, sumB, sumA2, sumB2)
    return ip, corr, n

def compute_correlation(ip, n, sumA, sumB, sumA2, sumB2):
    if n==0:
        return np.nan
    meanA = sumA/n
    meanB = sumB/n
    numerator = ip - n*meanA*meanB
    denominator = np.sqrt(sumA2-(n*(meanA**2))) * np.sqrt(sumB2-(n*(meanB**2)))
    corr = numerator/denominator
    return corr

def get_l2_norm(vecA, vecB):
    iA = np.array([1 if i!=0 else 0 for i in vecA])
    iB = np.array([1 if i!=0 else 0 for i in vecB])
    vecA2 = vecA**2
    vecB2 = vecB**2

    vecA_l2 = np.linalg.norm(vecA, ord=2)
    vecB_l2 = np.linalg.norm(vecB, ord=2)
    iA_l2 = np.linalg.norm(iA, ord=2)
    iB_l2 = np.linalg.norm(iB, ord=2)
    vecA2_l2 = np.linalg.norm(vecA2, ord=2)
    vecB2_l2 = np.linalg.norm(vecB2, ord=2)

    return vecA_l2, vecB_l2, iA_l2, iB_l2, vecA2_l2, vecB2_l2

def get_scale(vecA, vecB):
    vecA_l2, vecB_l2, iA_l2, iB_l2, vecA2_l2, vecB2_l2 = get_l2_norm(vecA, vecB)
    ip_scale = vecA_l2*vecB_l2
    join_size_scale = iA_l2*iB_l2
    sumA_scale = vecA_l2*iB_l2
    sumB_scale = iA_l2*vecB_l2
    sumA2_scale = vecA2_l2*iB_l2
    sumB2_scale = iA_l2*vecB2_l2
    return (ip_scale, join_size_scale, 
        sumA_scale, sumB_scale, 
        sumA2_scale, sumB2_scale)

def compute_sample_size(t, mode, storage_size):
    if mode == "join_size":
        jl_sample_size = mh_sample_size = kmv_sample_size = threshold_sample_size = priority_sample_size = wmh_sample_size = storage_size
        cs_sample_size = int(storage_size/(2*t-1))
    elif mode == "ip" or mode == "time":
        jl_sample_size = int(storage_size/1)
        cs_sample_size = int(storage_size/(2*t-1))
        mh_sample_size = kmv_sample_size = wmh_sample_size = threshold_sample_size = priority_sample_size = int(storage_size/1.5)
    elif mode == "corr":
        jl_sample_size = int(storage_size/3)
        cs_sample_size = int(storage_size/(3*(2*t-1)))
        mh_sample_size = kmv_sample_size = threshold_sample_size = int(storage_size/1.5)
        priority_sample_size = int(storage_size/1.5)
        wmh_sample_size = int(storage_size/4)
    return wmh_sample_size,kmv_sample_size,mh_sample_size,jl_sample_size,cs_sample_size,priority_sample_size,threshold_sample_size


def get_sketcher(wmh_sample_size, kmv_sample_size, mh_sample_size, jl_sample_size, cs_sample_size, priority_sample_size, threshold_sample_size, sketch_method, i_seed):
    if sketch_method == 'jl': sketcher = JL(jl_sample_size, i_seed)
    elif sketch_method == 'cs': sketcher = CS(cs_sample_size, i_seed)
    elif sketch_method == 'kmv': sketcher = KMV(kmv_sample_size, i_seed)
    elif sketch_method == 'mh': sketcher = MH(mh_sample_size, i_seed)
    elif sketch_method == 'wmh': sketcher = WMH(wmh_sample_size, i_seed)
    elif sketch_method == 'dmh': sketcher = DartMH(wmh_sample_size, i_seed)
    elif sketch_method == 'ps_2norm': sketcher = PS(priority_sample_size, i_seed, norm=2)
    elif sketch_method == 'ps_1norm': sketcher = PS(priority_sample_size, i_seed, norm=1)
    elif sketch_method == 'ps_uniform': sketcher = PS(priority_sample_size, i_seed, norm=0)
    elif sketch_method == 'ts_2norm': sketcher = TS(threshold_sample_size, i_seed, norm=2)
    elif sketch_method == 'ts_1norm': sketcher = TS(priority_sample_size, i_seed, norm=1)
    elif sketch_method == 'ts_uniform': sketcher = TS(priority_sample_size, i_seed, norm=0)
    elif sketch_method == 'ts_corr': sketcher = TSCorr(threshold_sample_size, i_seed)
    elif sketch_method == 'ps_corr': sketcher = PSCorr(priority_sample_size, i_seed)
    elif sketch_method == 'ts_corr012': sketcher = TS012Corr(threshold_sample_size, i_seed)
    elif sketch_method == 'ps_corr012': sketcher = PS012Corr(priority_sample_size, i_seed)
    return sketcher


plot_parameters = {
    'jl': ('JL', 's', '#dc267f', 'dashed', 1.0, 1.0, 1.0),
    'cs': ('CS', 's', '#dc267f', 'solid', 1.0, 1.0, 1.0),
    'kmv': ('KMV-NumPy', 's', '#785ef0', 'dotted', 1.0, 1.0, 1.0),
    'mh': ('MH', 's', '#fe6100', 'dashed', 1.0, 1.0, 1.0),
    'wmh': ('MH-weighted', 's', '#fe6100', 'solid', 1.0, 1.0, 1.0),
    'dmh': ('DartMH', '*', '#fe6100', 'solid', 1.0, 1.0, 1.0),
    'ts_2norm': ('TS-weighted', 's', '#ffb000', 'solid', 1.0, 1.0, 1.0),
    'ts_1norm': ('TS-1norm', 's', '#ffb000', 'dotted', 1.0, 1.0, 1.0),
    'ts_uniform': ('TS-uniform', 's', '#ffb000', 'dashed', 1.0, 1.0, 1.0),
    'ps_2norm': ('PS-weighted', 's', '#785ef0', 'solid', 1.0, 1.0, 1.0),
    'ps_1norm': ('PS-1norm', 's', '#785ef0', 'dotted', 1.0, 1.0, 1.0),
    'ps_uniform': ('PS-uniform', 's', '#785ef0', 'dashed', 1.0, 1.0, 1.0),
    'ts_corr': ('TS-weighted', 's', '#ffb000', 'solid', 1.0, 1.0, 1.0),
    'ps_corr': ('PS-weighted', 's', '#785ef0', 'solid', 1.0, 1.0, 1.0),
    'ts_corr012': ('TS-l0l1l2', 's', '#ffb000', 'dashed', 1.0, 1.0, 1.0),
    'ps_corr012': ('PS-l0l1l2', 's', '#785ef0', 'dashed', 1.0, 1.0, 1.0),
}