import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import sys
import os
sys.path.append(os.getenv("PROJECT_PATH"))
from utils import true_values, get_sketcher, compute_sample_size, plot_parameters
from script.experiment_corr import compute_estimation as compute_estimation_corr
from script.experiment_ip import compute_estimation as compute_estimation_ip
import argparse
from sklearn.metrics import r2_score

#################### Main ####################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", "--mode", 
        help="mode", type=str)
    args = parser.parse_args()
    mode = args.mode

    if mode == "join_size":
        data = pickle.load(open("log/wbf_data_joinSize", "rb"))
    else:
        data = pickle.load(open("log/wbf_data", "rb"))
    storage_sizes = [400]
    iteration = 1
    t = 1

    if mode == "corr":
        sketch_methods = ['jl', 'cs', 'mh', 'wmh', 'ts_corr', 'ps_corr', 'ts_uniform', 'ps_uniform']
    else:
        sketch_methods = ['jl', 'cs', 'mh', 'wmh', 'ts_2norm', 'ps_2norm', 'ts_uniform', 'ps_uniform']

    results = defaultdict(dict)
    log_name = 'log/analysis_worldBank_' + mode
    for enum,log_key in enumerate(data):
        print(f"🚀🚀🚀 {enum} of {len(data)}")
        vecA = data[log_key]['vecA']
        vecB = data[log_key]['vecB']
        if len(vecA) < 400 or len(vecB) < 400: # Skip if the length of the vectors is less than 400, not meaningful to do sketching
            continue
        if mode=="corr":
            iA = np.array([1 if i != 0 else 0 for i in vecA])
            iB = np.array([1 if i != 0 else 0 for i in vecB])
            vecA2 = vecA ** 2
            vecB2 = vecB ** 2
        for storage_size in storage_sizes:
            for i_iter in range(iteration):
                print("storage_size:", storage_size, "iteration:", i_iter)
                wmh_sample_size, kmv_sample_size, mh_sample_size, jl_sample_size, cs_sample_size, priority_sample_size, threshold_sample_size = compute_sample_size(t, mode, storage_size)
                # True Values
                ip, corr, n = true_values(vecA, vecB)
                if (np.isnan(corr) or np.isinf(corr)) and (mode=="corr"):
                    continue
                results[log_key]['true'] = (ip, corr, n)

                for sketch_method in sketch_methods:
                    sketcher = get_sketcher(wmh_sample_size, kmv_sample_size, mh_sample_size, jl_sample_size, cs_sample_size, priority_sample_size, threshold_sample_size, sketch_method, t, i_iter)
                    if mode=='corr':
                        sk_a, sk_b, corr_est = compute_estimation_corr(vecA, vecB, iA, iB, vecA2, vecB2, sketch_method, sketcher)
                        if 'ps' in sketch_method or 'ts' in sketch_method:
                            mem_sizeA = sk_a.sk_values.shape[0]
                            mem_sizeB = sk_b.sk_values.shape[0]
                            results[log_key][sketch_method] = (corr_est, mem_sizeA*1.5, mem_sizeB*1.5)
                        else:
                            results[log_key][sketch_method] = corr_est
                    else:
                        sk_a, sk_b, _, ip_est, _ = compute_estimation_ip(vecA, vecB, sketcher)
                        if 'ps' in sketch_method or 'ts' in sketch_method:
                            mem_sizeA = sk_a.sk_values.shape[0]
                            mem_sizeB = sk_b.sk_values.shape[0]
                            results[log_key][sketch_method] = (ip_est, mem_sizeA*1.5, mem_sizeB*1.5)
                        else:
                            results[log_key][sketch_method] = ip_est
                pickle.dump(results, open(log_name, "wb"))


    # make plot
    data_list = defaultdict(list)
    for log_key in results:
        ip, corr, n = results[log_key]['true']
        if np.isnan(corr) or np.isinf(corr):
            continue
        for sketch_method in sketch_methods:
            if mode!='corr':
                try:
                    ip_est, _, _ = results[log_key][sketch_method]
                except:
                    ip_est = results[log_key][sketch_method]
                data_list[sketch_method].append((ip, ip_est))
            else:
                try:
                    corr_est, _, _ = results[log_key][sketch_method]
                except:
                    corr_est = results[log_key][sketch_method]
                if np.isnan(corr_est) or np.isinf(corr_est):
                    continue
                if corr_est>1:
                    corr_est=1
                if corr_est<-1:
                    corr_est=-1
                data_list[sketch_method].append((corr, corr_est))
    
    avg_diffs, r2_scores = [], []
    for sketch_method in sketch_methods:
        res = data_list[sketch_method]
        val_t = [i[0] for i in res]
        val_e = [i[1] for i in res]
        val_diff = np.average([abs(i[0]-i[1]) for i in res])
        r2 = r2_score(val_t, val_e)
        avg_diffs.append(round(val_diff, 3))
        r2_scores.append(round(r2, 3))
    data = {"Sketch Method":[plot_parameters[sketch_method][0] for sketch_method in sketch_methods], "Average Difference":avg_diffs, "R2 Score":r2_scores}
    df = pd.DataFrame(data)
    sorted_df = df.sort_values(by="Average Difference", ascending=True)
    sorted_df.to_csv('log/WorldBank_' + mode+'.csv', index=False)

    if mode=='corr':
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size': 16})
        sketch_methods_plot = ['ps_corr', 'jl', 'cs']

        for sketch_method in sketch_methods_plot:
            res = data_list[sketch_method]
            corr_t = [i[0] for i in res]
            corr_e = [i[1] for i in res]
            corr_diff = np.average([abs(i[0]-i[1]) for i in res])
            r2 = r2_score(corr_t, corr_e)
            print(f"sketch_method: {sketch_method}\ncorr_diff: {corr_diff}\nr2_score: {r2}")

            r2 = round(r2, 3)
            err = round(corr_diff, 3)
            plt.scatter(corr_t, corr_e, s=3, alpha=0.1,label=f'$R^2$: {r2}\nAvg error: {err}')
            plt.legend(loc='lower right')
            plt.xlabel('Actual Correlation', weight='bold')
            plt.ylabel('Estimated Correlation', weight='bold')
            plt.xlim(-0.7)
            plt.savefig('fig/wbf_corr_scatter_'+plot_parameters[sketch_method][0]+'.pdf', bbox_inches='tight')
            plt.close()
    