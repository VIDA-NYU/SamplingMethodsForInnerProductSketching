import pickle
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import argparse
import os
import sys
project_path = os.getenv("PROJECT_PATH")
sys.path.append(os.getenv("PROJECT_PATH"))
from utils import get_scale, plot_parameters

def generate_plot_data(log_results, sketch_method, mode='ip'):
    plot_data = defaultdict(lambda: defaultdict(list))
    vecA, vecB = log_results['vecA'], log_results['vecB']
    (ip_scale, join_size_scale, _, _, _, _) = get_scale(vecA, vecB)
    for key in log_results:
        if key in ['vecA', 'vecB']:
            continue
        sample_size, _, _ = key.split("_")
        sample_size = int(sample_size)
        ip, corr, join_size = log_results[key]['true']
        if mode in ['ip', 'join', 'time']:
            try:
                ip_est, sketch_time, est_time = log_results[key][sketch_method]
            except:
                ip_est, sketch_time, est_time, mem_sizeA, mem_sizeB = log_results[key][sketch_method]
                try:
                    ip_est, cnt = ip_est
                    plot_data['mem_sizeA'][sample_size].append(cnt)
                    plot_data['mem_sizeB'][sample_size].append(cnt)
                except:
                    ip_est, cnt = ip_est, 0
            ip_diff = abs(ip_est - ip)/ip_scale
            plot_data['ip_diff'][sample_size].append(ip_diff)
            plot_data['sketch_time'][sample_size].append(sketch_time)
            plot_data['est_time'][sample_size].append(est_time)
        elif mode=='corr':
            corr_max_diff = min(max(abs(1-corr), abs(-1-corr)), 1)
            try:
                corr_est, mem_sizeA, mem_sizeB = log_results[key][sketch_method]
                plot_data['mem_sizeA'][sample_size].append(mem_sizeA)
                plot_data['mem_sizeB'][sample_size].append(mem_sizeB)
            except:
                corr_est = log_results[key][sketch_method]
            if np.isnan(corr_est) or np.isinf(corr_est):
                corr_diff = corr_max_diff
                plot_data['corr_diff'][sample_size].append(corr_diff)
                continue
            else:
                if corr_est>1:
                    corr_est=1
                if corr_est<-1:
                    corr_est=-1
            corr_diff = abs(corr_est - corr)
            if corr_diff > corr_max_diff:
                corr_diff = corr_max_diff
            plot_data['corr_diff'][sample_size].append(corr_diff)
    plot_data_std = {key: {s: np.std(data[s]) for s in data} for key, data in plot_data.items()}
    plot_data = {key: {s: np.mean(data[s]) for s in data} for key, data in plot_data.items()}
    plot_est = {k:str(round(s*1000, 4))+'ms' for k,s in plot_data_std['est_time'].items() if k!=100}
    plot_est = {k:str(round(s*1000, 4))+'ms' for k,s in plot_data_std['est_time'].items() if k!=100}
    print(f"sketch_method:{sketch_method}\n STDev of est_time on different sketch size: {plot_est}")
    return plot_data


def make_plot(plot_data, sketch_methods, plot_parameters, plot_type='ip_diff', title=None, fig_loc=None):
    print("="*33)
    print(f'plot_type: {plot_type}')
    plt.rcParams.update({'font.size': 16})
    for sketch_method in sketch_methods:
        x, y = get_x_y(plot_data, plot_type, sketch_method)
        alpha = 0.99
        plt.plot(x, y, alpha=alpha, linewidth=1.8,
                    label=plot_parameters[sketch_method][0], 
                    marker=plot_parameters[sketch_method][1], 
                    color=plot_parameters[sketch_method][2],
                    linestyle=plot_parameters[sketch_method][3])
    # plt.legend(loc='upper center', 
    #     bbox_to_anchor=(0.45, 1.35),
    #     ncol=4)
    plt.ylabel("Scaled Average Difference", weight='bold')
    plt.xlabel("Storage Size", weight='bold')
    plt.ylim(bottom=-0.003)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) 
    if plot_type == 'corr_diff':
        plt.ylim(bottom=-0.02)
        plt.ylabel("Average Difference", weight='bold')
    elif 'time' in plot_type:
        if plot_type == 'est_time':
            plt.ylim(bottom=-0.005)
        if plot_type == 'sketch_time':
            plt.ylim(bottom=-0.5)
        plt.ylabel("Average Time (Seconds)", weight='bold')
        plt.xlabel("Sketch Size", weight='bold')
    if title:
        plt.title(title)
    if fig_loc:
        # plt.savefig(fig_loc, bbox_inches='tight')
        plt.savefig(fig_loc, format='pdf', dpi=None, bbox_inches="tight")
    plt.close()

def get_x_y(plot_data, plot_type, sketch_method):
    y = list(plot_data[sketch_method][plot_type].values())
    print("="*33)
    if 'time' in plot_type:
        y = y[1:]
        if plot_type == 'est_time':
            y = [i*1000 for i in y]
            if sketch_method == 'ps_2norm':
                print(f'sketch_method: {sketch_method}\n{plot_type}: {" & ".join([f"{i:.3f}ms" for i in y])}')
            else:
                print(f'sketch_method: {sketch_method}\n{plot_type}: {" & ".join([f"{i:.3f}ms" for i in y])}')
        else:
            print(f'sketch_method: {sketch_method}\n{plot_type}: {" & ".join([f"{i:.3f}s" for i in y])}')
        ip_diff = list(plot_data[sketch_method]['ip_diff'].values())[1:]
        print(f'ip_diff: {", ".join([f"{i:.6f}" for i in ip_diff])}')
        x = list(plot_data[sketch_method][plot_type].keys())[1:]
        print(f'x: {", ".join([f"{i:.2f}" for i in x])}')
        if 'mem_sizeA' in plot_data[sketch_method]:
            mA = list(plot_data[sketch_method]['mem_sizeA'].values())
            mB = list(plot_data[sketch_method]['mem_sizeB'].values())
            xx = [(i+j)/2 for i,j in zip (mA, mB)]
            print(f'mem_size: {", ".join([f"{i:.2f}" for i in xx])}')
    else:
        print(f'sketch_method: {sketch_method}\n{plot_type}: {", ".join([f"{i:.4f}" for i in y])}')
        if 'mem_sizeA' in plot_data[sketch_method]:
            mA = list(plot_data[sketch_method]['mem_sizeA'].values())
            mB = list(plot_data[sketch_method]['mem_sizeB'].values())
            x = [(i+j)/2 for i,j in zip (mA, mB)]
            print(f'mem_size: {", ".join([f"{i:.2f}" for i in x])}')
        else:
            x = list(plot_data[sketch_method][plot_type].keys())
            print(f'x: {", ".join([f"{i:.2f}" for i in x])}')
    return x, y


def extract_from(data_file):
    log_file_name = data_file.split("/")[-1]
    name_splits = log_file_name.split("+")
    print("name_splits", name_splits)
    # Extract the values
    overlap = float(name_splits[0].split('_')[1])
    outlier = float(name_splits[1].split('_')[1])
    corr = float(name_splits[2].split('_')[1])
    mode = name_splits[3].split('_')[1]
    title_suffix = str(overlap)+" Overlap, "+str(outlier)+" Outlier"
    return overlap,outlier,corr,mode,title_suffix,log_file_name

def create_plot_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_file", "--data_file",
        help="data file to make plots", type=str)
    parser.add_argument("-sketch_methods", "--sketch_methods",
        help="sketch methods to run", type=str)
    args = parser.parse_args()
    assert args.sketch_methods is not None, "sketch_methods is missing"
    data_file = args.data_file
    sketch_methods = args.sketch_methods.split("+")
    return data_file, sketch_methods

if __name__ == "__main__":
    data_file, sketch_methods = create_plot_parser()
    overlap, outlier, corr, mode, title_suffix, log_file_name = extract_from(data_file)
    log_results = pickle.load(open(data_file, "rb"))
    print(f"mode: {mode}, sketch_methods: {sketch_methods}")
    
    if mode == 'ip' or mode == 'join':
        plot_types = ['ip_diff']
    elif mode == 'corr':
        title_suffix = str(corr)+" Corr, "+title_suffix
        plot_types = ['corr_diff']
    elif mode == 'time':
        plot_types = ['sketch_time', 'est_time']

    plot_data = {sketch_method:generate_plot_data(log_results, sketch_method, mode=mode) for sketch_method in sketch_methods}
    for plot_type in plot_types:
        make_plot(plot_data, sketch_methods, plot_parameters, 
            plot_type=plot_type, 
            # title = title_suffix,
            fig_loc=project_path+'/fig/'+plot_type+'-'+log_file_name+'.pdf')