import pickle
from collections import defaultdict
import sys
import os
sys.path.append(os.getenv("PROJECT_PATH"))
from script.plot import generate_plot_data, make_plot
from script.experiment_ip import compute_estimation
from utils import true_values, compute_sample_size, get_sketcher, get_scale, plot_parameters

#################### Main ####################
if __name__ == "__main__":
    data = pickle.load(open("log/twitter_data", "rb"))
    sketch_methods = ['jl', 'cs', 'mh', 'wmh', 'ts_2norm', 'ts_uniform', 'ps_2norm', 'ps_uniform']
    storage_sizes = [i for i in range(100, 1001, 200)]
    iteration = 100
    t = 1
    mode = 'ip'
    for key in data:
        vecA = data[key]['vecA']
        vecB = data[key]['vecB']
        log_name = 'log/analysis_' + key
        results = defaultdict(dict)
        (ip_scale, _, _, _, _, _) = get_scale(vecA, vecB)
        for storage_size in storage_sizes:
            for i_iter in range(iteration):
                print(f"storage_size: {storage_size}, iteration: {i_iter} of {iteration}")
                log_key = str(storage_size)+'_'+str('NA')+'_'+str(i_iter)
                wmh_sample_size, kmv_sample_size, mh_sample_size, jl_sample_size, cs_sample_size, priority_sample_size, threshold_sample_size = compute_sample_size(t, mode, storage_size)
                # True Values
                ip, corr, n = true_values(vecA, vecB)
                results[log_key]['true'] = (ip, corr, n, ip_scale)
                for sketch_method in sketch_methods:
                    sketcher = get_sketcher(wmh_sample_size, kmv_sample_size, mh_sample_size, jl_sample_size, cs_sample_size, priority_sample_size, threshold_sample_size, sketch_method, t, i_iter)
                    sk_a, sk_b, sketch_time, ip_est, est_time = compute_estimation(vecA, vecB, sketcher)
                    if 'ps' in sketch_method or 'ts' in sketch_method:
                        mem_sizeA = sk_a.sk_values.shape[0]
                        mem_sizeB = sk_b.sk_values.shape[0]
                        results[log_key][sketch_method] = (ip_est, sketch_time, est_time, mem_sizeA*1.5, mem_sizeB*1.5)
                    else:
                        results[log_key][sketch_method] = (ip_est, sketch_time, est_time)
                pickle.dump(results, open(log_name, "wb"))

        plot_data = {sketch_method:generate_plot_data(results, sketch_method, mode=mode) for sketch_method in sketch_methods}
        plot_type = 'ip_diff'
        make_plot(plot_data, sketch_methods, plot_parameters, 
                    plot_type=plot_type, 
                    fig_loc='fig/analysis_twitter.pdf'
                )
