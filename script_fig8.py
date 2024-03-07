import pickle
import numpy as np
from collections import defaultdict
import sys
import os
sys.path.append(os.getenv("PROJECT_PATH"))
from utils import true_values, get_sketcher, compute_sample_size
from script.experiment_corr import compute_estimation as compute_estimation_corr
from script.experiment_ip import compute_estimation as compute_estimation_ip
import time
import argparse

#################### Main ####################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", "--mode", 
        help="mode", type=str)
    args = parser.parse_args()
    mode = args.mode

    if mode == "join_size":
        data = pickle.load(open("log/vecAvecB_202312122036", "rb"))
    else:
        data = pickle.load(open("log/vecAvecB_202308101818", "rb"))
    storage_sizes = [400]
    iteration = 1
    t = 1
    results_compare = None
    # if mode == "corr":
        # results_compare = pickle.load(open("log/analysis_corr_202309261753", "rb"))
    # elif mode == "ip":
        # results_compare = pickle.load(open("log/analysis_ip_202308161621", "rb"))
    # elif mode == "join_size":
    #     results_compare = pickle.load(open("log/analysis_join_size_202312130038", "rb"))

    if mode == "corr":
        sketch_methods = ['jl', 'cs', 'kmv', 'mh', 'wmh', 'ts_uniform', 'ts_corr', 'ps_uniform', 'ps_corr']
        sketch_methods = ['jl', 'cs', 'mh', 'wmh', 'ts_corr', 'ps_corr', 'ts_1norm', 'ps_1norm', 'ts_uniform', 'ps_uniform']
    elif mode == "ip":
        # sketch_methods = ['jl', 'cs', 'mh', 'wmh', 'ts_2norm', 'ts_uniform', 'ps_2norm', 'ps_uniform']
        sketch_methods = ['jl', 'cs', 'mh', 'wmh', 'ts_2norm', 'ps_2norm', 'ts_1norm', 'ps_1norm', 'ts_uniform', 'ps_uniform']
    elif mode == "join_size":
        sketch_methods = ['jl', 'cs', 'mh', 'wmh', 'ts_2norm', 'ps_2norm', 'ts_1norm', 'ps_1norm', 'ts_uniform', 'ps_uniform']

    results = defaultdict(dict)
    current_time = time.strftime("%Y,%m,%d,%H,%M,%S").split(',')
    log_suffix = ''.join(current_time[:-1])
    log_name = 'log/analysis_' + mode + '_size_' + str(storage_sizes[0]) + '_' + log_suffix

    for enum,log_key in enumerate(data):
        print(f"🚀🚀🚀 {enum} of {len(data)}")
        # if enum in [135, 161]:
        #     print("skip")
        #     continue
        vecA = data[log_key]['vecA']
        vecB = data[log_key]['vecB']
        if len(vecA) < 400 or len(vecB) < 400:
            continue
        if mode=="corr":
            iA = np.array([1 if i != 0 else 0 for i in vecA])
            iB = np.array([1 if i != 0 else 0 for i in vecB])
            vecA2 = vecA ** 2
            vecB2 = vecB ** 2
        for storage_size in storage_sizes:
            for i_iter in range(iteration):
                wmh_sample_size, kmv_sample_size, mh_sample_size, jl_sample_size, cs_sample_size, priority_sample_size, threshold_sample_size = compute_sample_size(t, mode, storage_size)
                # True Values
                ip, corr, n = true_values(vecA, vecB)
                if (np.isnan(corr) or np.isinf(corr)) and (mode=="corr"):
                    continue
                results[log_key]['true'] = (ip, corr, n)
                for sketch_method in sketch_methods:
                    # if results_compare is not None and log_key in results_compare and sketch_method in results_compare[log_key] and sketch_method!='cs':
                    if results_compare is not None and log_key in results_compare and sketch_method in results_compare[log_key] and sketch_method not in ['cs']:
                        results[log_key][sketch_method] = results_compare[log_key][sketch_method]
                        print(f"sketch_method in results_compare: {sketch_method}")
                        continue
                    print("+"*20, sketch_method, "+"*20)
                    sketcher = get_sketcher(wmh_sample_size, kmv_sample_size, mh_sample_size, jl_sample_size, cs_sample_size, priority_sample_size, threshold_sample_size, sketch_method, t, i_iter)
                    if mode=='corr':
                        sk_a, sk_b, corr_est = compute_estimation_corr(vecA, vecB, iA, iB, vecA2, vecB2, sketch_method, sketcher)
                        print(f"sketch_method: {sketch_method}\nstorage_size: {storage_size}\nsketch_size: {sketcher.sketch_size}")
                        print(f"corr_est: {corr_est}\ntrue_corr: {corr}")
                        if 'ps' in sketch_method or 'ts' in sketch_method:
                            mem_sizeA = sk_a.sk_values.shape[0]
                            mem_sizeB = sk_b.sk_values.shape[0]
                            results[log_key][sketch_method] = (corr_est, mem_sizeA*1.5, mem_sizeB*1.5)
                            print("mem_sizeA: %s\nmem_sizeB: %s" % (mem_sizeA, mem_sizeB))
                        else:
                            results[log_key][sketch_method] = corr_est
                    elif mode=='ip' or mode=='join_size':
                        sk_a, sk_b, sketch_time, ip_est, est_time = compute_estimation_ip(vecA, vecB, sketcher)
                        print(f"sketch_method: {sketch_method}\nstorage_size: {storage_size}\nsketch_size: {sketcher.sketch_size}")
                        print(f"sketch_time: {sketch_time}\nest_time: {est_time}")
                        print(f"ip_est: {ip_est}\ntrue_ip: {ip}")
                        if 'ps' in sketch_method or 'ts' in sketch_method:
                            mem_sizeA = sk_a.sk_values.shape[0]
                            mem_sizeB = sk_b.sk_values.shape[0]
                            results[log_key][sketch_method] = (ip_est, sketch_time, est_time, mem_sizeA*1.5, mem_sizeB*1.5)
                            print("mem_sizeA: %s\nmem_sizeB: %s" % (mem_sizeA, mem_sizeB))
                        else:
                            results[log_key][sketch_method] = (ip_est, sketch_time, est_time)
                print("len(results):", len(results))
                pickle.dump(results, open(log_name, "wb"))