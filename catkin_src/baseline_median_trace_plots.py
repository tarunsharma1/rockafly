import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob


################ this code is incomplete #################

subdir = 'UXJ88yawpos/'
all_flies = glob.glob('/home/tarun/catkin_ws/src/trajectories_autostep_ros/bagfiles/hdf5/'+subdir+'/*')
pickle_files = []

for i in all_flies:
	pickle_files.append(glob.glob('/home/tarun/catkin_ws/src/trajectories_autostep_ros/bagfiles/hdf5/'+subdir+i.split('/')[-1]+'/avg_left_minus_right*.p'))
	

print (pickle_files)
##pickle_files = [['/home/tarun/catkin_ws/src/trajectories_autostep_ros/bagfiles/hdf5/UX-S00-top-view/Nov-2-fly5/avg_left_minus_right_Nov-2-fly5.p'], ['/home/tarun/catkin_ws/src/trajectories_autostep_ros/bagfiles/hdf5/UX-S00-top-view/Nov-2-fly7/avg_left_minus_right_Nov-2-fly7.p'], ['/home/tarun/catkin_ws/src/trajectories_autostep_ros/bagfiles/hdf5/UX-S00-top-view/Nov-1-fly1/avg_left_minus_right_Nov-1-fly1.p'], ['/home/tarun/catkin_ws/src/trajectories_autostep_ros/bagfiles/hdf5/UX-S00-top-view/Nov-1-fly2/avg_left_minus_right_Nov-1-fly2.p'], ['/home/tarun/catkin_ws/src/trajectories_autostep_ros/bagfiles/hdf5/UX-S00-top-view/Nov-2-fly6/avg_left_minus_right_Nov-2-fly6.p'], ['/home/tarun/catkin_ws/src/trajectories_autostep_ros/bagfiles/hdf5/UX-S00-top-view/Nov-2-fly4/avg_left_minus_right_Nov-2-fly4.p'], ['/home/tarun/catkin_ws/src/trajectories_autostep_ros/bagfiles/hdf5/UX-S00-top-view/Nov-2-fly3/avg_left_minus_right_Nov-2-fly3.p']]

## only do baseline stuff for first baseline
num_speeds = 2
trajectories = []
trajectories_time = []

mean_baseline_left_trace = {}
mean_baseline_right_trace = {}


for i in range(0, num_speeds):
	mean_baseline_trace = []

	mean_baseline_left_trace[str(i)] = []
	mean_baseline_right_trace[str(i)] = []

	
	# will be used for resampling
	time_and_WBA_for_all_flies = []

	for f in pickle_files:
		# skip empty entries in list
		if f==[]:
			continue
		dict_for_pickle = pickle.load(open(f[0], 'rb'))

		## each key for the dict is the speed..the value is a list of [t, WBA]
		## ok so for a speed, the time for each fly starts at 0 and ends at the exact same (to first decimal) value
		## the number of points in time (and hence WBA) are different though fly to fly..so will have to resample

		time = dict_for_pickle[i][0]

		if 'trajectory_1' and 'trajectory_1_time' in dict_for_pickle.keys():
			for x in range(0,num_speeds):
				trajectories.append(dict_for_pickle['trajectory_'+str(x)][0])
				trajectories_time.append(dict_for_pickle['trajectory_'+str(x)+'_time'][0])

		if 'mean_baseline_left_trace_1' in dict_for_pickle.keys():
			mean_baseline_left_trace[str(i)].append(dict_for_pickle['mean_baseline_left_trace_'+str(i)])
			mean_baseline_right_trace[str(i)].append(dict_for_pickle['mean_baseline_right_trace_'+str(i)])
			## L + R
			mean_baseline_trace.append(dict_for_pickle['mean_baseline_left_trace_'+str(i)] + dict_for_pickle['mean_baseline_right_trace_'+str(i)])

	## resample all the baselines (L + R) to have the same time
	import ipdb;ipdb.set_trace()


