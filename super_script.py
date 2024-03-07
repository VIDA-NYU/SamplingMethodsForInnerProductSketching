import os
import time
import argparse
import sys

project_path = os.getenv("PROJECT_PATH")
script_path = os.getenv("SCRIPT_PATH")

def commoand_plot(sketch_methods, data_file, fig_name=None):
	log_time = data_file.split("+")[-1]
	if fig_name is not None:
		command = 'time python \
			'+script_path+'/plot.py \
			-data_file='+data_file+' \
			-sketch_methods='+str(sketch_methods)+' \
			-fig_name='+fig_name+' \
			> '+project_path+'/debug_log/plot_data_mode_'+mode+'_'+log_time
	else:
		command = 'time python \
			'+script_path+'/plot.py \
			-data_file='+data_file+' \
			-sketch_methods='+str(sketch_methods)+' \
			> '+project_path+'/debug_log/plot_data_mode_'+mode+'_'+log_time
	print("🚀🚀🚀 running: plot")
	print(command)
	os.system(command)

def command_experiment_ip(outlier_pct, outlier_max, mode, sketch_methods, corr, t, start_size, end_size, interval_size, iteration, overlap, log_time, log_name):
    command = 'time python \
			'+script_path+'/experiment_ip.py \
			-overlap='+str(overlap)+' \
			-outlier='+str(outlier_pct)+' \
			-outlier_max='+str(outlier_max)+' \
			-start_size='+str(start_size)+' \
			-end_size='+str(end_size)+' \
			-interval_size='+str(interval_size)+' \
			-corr='+str(corr)+' \
			-t='+str(t)+' \
			-mode='+mode+' \
			-iteration='+str(iteration)+' \
			-log_name='+str(log_name)+' \
			-sketch_methods='+sketch_methods+' \
			> '+project_path+'/debug_log/run_mode_'+mode+'_'+log_time
    print("🚀🚀🚀 running: experiment")
    os.system(command)
    
def command_experiment_corr(outlier_pct, outlier_max, mode, sketch_methods, corr, t, start_size, end_size, interval_size, iteration, overlap, log_time, log_name):
    command = 'time python \
			'+script_path+'/experiment_corr.py \
			-overlap='+str(overlap)+' \
			-outlier='+str(outlier_pct)+' \
			-outlier_max='+str(outlier_max)+' \
			-start_size='+str(start_size)+' \
			-end_size='+str(end_size)+' \
			-interval_size='+str(interval_size)+' \
			-corr='+str(corr)+' \
			-t='+str(t)+' \
			-mode='+mode+' \
			-iteration='+str(iteration)+' \
			-log_name='+str(log_name)+' \
			-sketch_methods='+sketch_methods+' \
			> '+project_path+'/debug_log/run_mode_'+mode+'_'+log_time
    print("🚀🚀🚀 running: experiment")
    os.system(command)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-mode", "--mode", 
		help="mode", type=str)
	parser.add_argument("-plot_only", "--plot_only", 
		help="only plot", type=bool)
	parser.add_argument("--plot_names",
		nargs='+', help='Input list')
	# Check if the required parameter is present
	args = parser.parse_args()
	mode = args.mode or 'ip'
	plot_only = args.plot_only or False
	print("plot_only", plot_only)
	if plot_only:
		assert args.plot_names is not None, "plot_names is missing"
		plot_names = args.plot_names
		print("plot_names", plot_names)
	
	outlier_pcts = [0.02]
	outlier_maxes = [10] # min is default as 0
	overlaps = [0.01, 0.1, 0.5, 1.0][:]
	corrs = [-0.2, 0.4, -0.6, 0.8][:1]
	if mode=='ip':
		sketch_methods = ['jl', 'cs', 'mh', 'wmh', 'ts_uniform', 'ts_2norm', 'ps_uniform', 'ps_2norm']
	elif mode=='1normVS2norm':
		sketch_methods = ['ts_1norm', 'ts_2norm', 'ps_1norm', 'ps_2norm']
	elif mode == 'join_size':
		sketch_methods = ['jl', 'cs', 'mh', 'ts_uniform', 'ps_uniform']
	elif mode=='corr':
		sketch_methods = ['jl', 'cs', 'mh', 'wmh', 'ts_uniform', 'ts_corr', 'ps_uniform', 'ps_corr']
	elif mode=='time':
		sketch_methods = ['jl', 'cs', 'mh', 'dmh', 'ts_uniform', 'ts_2norm', 'ps_uniform', 'ps_2norm']
	sketch_methods = '+'.join(sketch_methods)

	if plot_only:
		for plot_name in plot_names:
			commoand_plot(sketch_methods, project_path+'/'+plot_name)
		sys.exit(0)

	if mode == 'ip' or mode == 'join_size':
		corr = 0.7
		start_size = 200
		end_size = 2000
		interval_size = 200
		iteration = 100
		t = 1
		for outlier_pct in outlier_pcts:
			for outlier_max in outlier_maxes:
				for overlap in overlaps:
					current_time = time.strftime("%Y,%m,%d,%H,%M,%S").split(',')
					log_time = ''.join(current_time)
					log_file_name = '+'.join(['mode_'+mode, 'overlap_'+str(overlap), 'outlier_'+str(outlier_pct), 'max_'+str(outlier_max), 'corr_'+str(corr), log_time])
					log_name = project_path+'/log/'+log_file_name
					# run experiment
					command_experiment_ip(outlier_pct, outlier_max, mode, sketch_methods, corr, t, start_size, end_size, interval_size, iteration, overlap, log_time, log_name)
					# plot
					commoand_plot(sketch_methods, log_name)
	elif mode == '1normVS2norm':
		corr = 0.7
		start_size = 200
		end_size = 2000
		interval_size = 200
		iteration = 100
		t = 1
		mode = 'ip'
		for outlier_pct in outlier_pcts:
			for outlier_max in outlier_maxes:
				for overlap in overlaps:
					current_time = time.strftime("%Y,%m,%d,%H,%M,%S").split(',')
					log_time = ''.join(current_time)
					log_file_name = '+'.join(['mode_'+mode, 'overlap_'+str(overlap), 'outlier_'+str(outlier_pct), 'max_'+str(outlier_max), 'corr_'+str(corr), log_time])
					log_name = project_path+'/log/'+log_file_name
					# run experiment
					# command_experiment_ip(outlier_pct, outlier_max, mode, sketch_methods, corr, t, start_size, end_size, interval_size, iteration, overlap, log_time, log_name)
					# plot
					# commoand_plot(sketch_methods, log_name)
					commoand_plot(sketch_methods, project_path+'/existing_log/mode_ip+overlap_'+str(overlap)+'+outlier_0.1+max_10+corr_0.7+fig5','fig5+overlap_'+str(overlap))
	elif mode == 'corr':
		overlap = 0.1
		start_size = 200
		end_size = 2000
		interval_size = 200
		iteration = 5
		t = 1
		for outlier_pct in outlier_pcts:
			for outlier_max in outlier_maxes:
				for corr in corrs:
					current_time = time.strftime("%Y,%m,%d,%H,%M,%S").split(',')
					log_time = ''.join(current_time)
					log_file_name = '+'.join(['mode_'+mode, 'overlap_'+str(overlap), 'outlier_'+str(outlier_pct), 'max_'+str(outlier_max), 'corr_'+str(corr), log_time])
					log_name = project_path+'/log/'+log_file_name
					# run experiment
					command_experiment_corr(outlier_pct, outlier_max, mode, sketch_methods, corr, t, start_size, end_size, interval_size, iteration, overlap, log_time, log_name)
					# plot
					commoand_plot(sketch_methods, log_name)
	elif mode == 'time':
		corr = 0.7
		t = 1
		start_size = 1000
		end_size = 5000
		interval_size = 1000
		iteration = 3
		overlap = 0.1
		outlier_pct = 0.1
		outlier_max = 10

		current_time = time.strftime("%Y,%m,%d,%H,%M,%S").split(',')
		log_time = ''.join(current_time)
		log_file_name = '+'.join(['mode_'+mode, 'overlap_'+str(overlap), 'outlier_'+str(outlier_pct), 'max_'+str(outlier_max), 'corr_'+str(corr), log_time])
		# log_file_name = '+'.join(['overlap_'+str(overlap), 'outlier_'+str(outlier_pct), 'corr_'+str(corr), 'mode_'+mode, log_time])
		log_name = project_path+'/log/'+log_file_name
		# run experiment
		command_experiment_ip(outlier_pct, outlier_max, mode, sketch_methods, corr, t, start_size, end_size, interval_size, iteration, overlap, log_time, log_name)
		# plot
		commoand_plot(sketch_methods, log_name)