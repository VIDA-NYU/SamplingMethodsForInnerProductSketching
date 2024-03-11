import os
import time
import argparse
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

project_path = os.getcwd()
script_path = os.path.join(project_path, "script")

sys.path.append(os.getenv("PROJECT_PATH"))
from script.plot import generate_plot_data, make_plot
from utils import plot_parameters

def commoand_plot(sketch_methods, data_file):
	log_time = data_file.split("+")[-1]
	command = 'time python \
			'+os.path.join(script_path, 'plot.py')+' \
			-data_file='+data_file+' \
			-sketch_methods='+str(sketch_methods)+' \
			> '+os.path.join(project_path, 'debug_log/plot_data_'+log_time)
	print("="*33)
	print("🚀🚀🚀 running: making plot")
	print(command)
	os.system(command)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# parser.add_argument("-mode", "--mode", 
	# 					help="mode of the experiment: ip or corr")
	# parser.add_argument("--data_files",
	# 					nargs='+', help='Input list of data files to plot')
	parser.add_argument("-paper_fig", "--paper_fig",
						help="specify the figure number in the paper you want to produce", type=int)
	args = parser.parse_args()
	# mode = args.mode or "ip"
	# data_files = args.data_files or None
	paper_fig = args.paper_fig or 3
	if paper_fig == 3:
		data_files = [
			"existing_log/mode_ip+overlap_0.01+outlier_0.02+max_10+corr_0.7+synthetic",
			"existing_log/mode_ip+overlap_0.1+outlier_0.02+max_10+corr_0.7+synthetic",
			"existing_log/mode_ip+overlap_0.5+outlier_0.02+max_10+corr_0.7+synthetic",
			"existing_log/mode_ip+overlap_1.0+outlier_0.02+max_10+corr_0.7+synthetic"
		]
		sketch_methods = ['jl', 'cs', 'mh', 'wmh', 'ts_uniform', 'ts_2norm', 'ps_uniform', 'ps_2norm']
	elif paper_fig == 4:
		data_files = [
			"existing_log/mode_join_size+overlap_0.01+outlier_0.02+max_10+corr_0.7+synthetic",
			"existing_log/mode_join_size+overlap_0.1+outlier_0.02+max_10+corr_0.7+synthetic",
			"existing_log/mode_join_size+overlap_0.5+outlier_0.02+max_10+corr_0.7+synthetic",
			"existing_log/mode_join_size+overlap_1.0+outlier_0.02+max_10+corr_0.7+synthetic"
		]
		sketch_methods = ['jl', 'cs', 'mh', 'ts_uniform', 'ps_uniform']
	elif paper_fig == 5:
		data_files = [
			"existing_log/mode_ip+overlap_0.01+outlier_0.1+max_10+corr_0.7+fig5",
			"existing_log/mode_ip+overlap_0.1+outlier_0.1+max_10+corr_0.7+fig5",
			"existing_log/mode_ip+overlap_0.5+outlier_0.1+max_10+corr_0.7+fig5",
			"existing_log/mode_ip+overlap_1.0+outlier_0.1+max_10+corr_0.7+fig5"
		]
		sketch_methods = ['ts_1norm', 'ts_2norm', 'ps_1norm', 'ps_2norm']
	elif paper_fig == 6:
		data_files = [
			"existing_log/mode_corr+overlap_0.1+outlier_0.02+max_10+corr_-0.2+synthetic",
			"existing_log/mode_corr+overlap_0.1+outlier_0.02+max_10+corr_0.4+synthetic",
			"existing_log/mode_corr+overlap_0.1+outlier_0.02+max_10+corr_0.6+synthetic",
			"existing_log/mode_corr+overlap_0.1+outlier_0.02+max_10+corr_0.8+synthetic"
		]
		sketch_methods = ['jl', 'cs', 'mh', 'wmh', 'ts_uniform', 'ts_corr', 'ps_uniform', 'ps_corr']
	elif paper_fig == 7:
		data_files = [
			"existing_log/mode_time+overlap_0.1+outlier_0.1+max_10+corr_0.7+synthetic"
		]
		sketch_methods = ['jl', 'cs', 'mh', 'dmh', 'ts_uniform', 'ts_2norm', 'ps_uniform', 'ps_2norm']
	elif paper_fig == 9:
		data = pickle.load(open("existing_log/20news_greater500", "rb"))
		sketch_methods = ['jl', 'cs', 'mh', 'wmh', 'ts_2norm', 'ts_uniform', 'ps_2norm', 'ps_uniform']
		plt.rcParams.update({'font.size': 16})
		x = [i for i in range(50,301,50)]
		for sketch_method in sketch_methods:
			y = data[sketch_method]['avg']
			plt.plot(x, y, 
					linestyle=plot_parameters[sketch_method][3],
					label=plot_parameters[sketch_method][0], 
					marker=plot_parameters[sketch_method][1], 
					color=plot_parameters[sketch_method][2])
		plt.legend(loc='upper center', 
			bbox_to_anchor=(0.45, 1.3),
			ncol=4)
		plt.ylim(bottom=-0.003)
		plt.xlabel('Storage Size', weight='bold')
		plt.ylabel('Average Difference', weight='bold')
		plt.savefig('fig/20news_greaterThan_500words.pdf', bbox_inches='tight')
		plt.close()

		data = pickle.load(open("existing_log/20news_all", "rb"))
		sketch_methods = ['jl', 'cs', 'mh', 'wmh', 'ts_2norm', 'ts_uniform', 'ps_2norm', 'ps_uniform']
		plt.rcParams.update({'font.size': 16})
		x = [i for i in range(50,301,50)]
		for sketch_method in sketch_methods:
			y = data[sketch_method]['avg']
			plt.plot(x, y, 
					linestyle=plot_parameters[sketch_method][3],
					label=plot_parameters[sketch_method][0], 
					marker=plot_parameters[sketch_method][1], 
					color=plot_parameters[sketch_method][2])
			
		plt.legend(loc='upper center', 
			bbox_to_anchor=(0.45, 1.3),
			ncol=4)
		plt.ylim(bottom=-0.003)
		plt.xlabel('Storage Size', weight='bold')
		plt.ylabel('Average Difference', weight='bold')
		plt.savefig('fig/20news_all.pdf', bbox_inches='tight')
		plt.close()

	elif paper_fig == 10:
		#tpch
		log_file_name = 'existing_log/analysis_supplier-lineitem-flat-z2-s1_iteration-300_storage-100-1000_tpch'
		log_results = pickle.load(open(log_file_name, "rb"))
		sketch_methods = ['jl', 'cs', 'mh', 'wmh', 'ts_uniform', 'ts_2norm', 'ps_uniform', 'ps_2norm']
		plot_data = {sketch_method:generate_plot_data(log_results, sketch_method, mode='ip') for sketch_method in sketch_methods}
		make_plot(plot_data, sketch_methods, plot_parameters, 
					plot_type='ip_diff', 
					fig_loc='fig/'+log_file_name.split('/')[1]+'.pdf'
				)

		#twitter
		log_file_name = "existing_log/analysis_follower-followee_twitter"
		log_results = pickle.load(open(log_file_name, "rb"))
		sketch_methods = ['jl', 'cs', 'mh', 'wmh', 'ts_2norm', 'ps_2norm', 'ts_uniform', 'ps_uniform']
		plot_data = {sketch_method:generate_plot_data(log_results, sketch_method, mode='ip') for sketch_method in sketch_methods}
		make_plot(plot_data, sketch_methods, plot_parameters, 
					plot_type='ip_diff', 
					fig_loc='fig/'+log_file_name.split('/')[1]+'.pdf'
				)

	sketch_methods = '+'.join(sketch_methods)

	if paper_fig not in [9,10]:
		for data_file in data_files:
			commoand_plot(sketch_methods, data_file)
