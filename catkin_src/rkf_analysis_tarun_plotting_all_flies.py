import matplotlib.pyplot as plt
import random
import numpy as np
import pickle
import glob
import scipy.interpolate as spi
import scipy.fftpack
import scipy.signal as sps
from scipy.optimize import leastsq
import seaborn as sns
import matplotlib as mpl



########################
# this script reads the saved pickle files for flies on a genotype (one per fly averaged over 3 trials), 
# resamples them to a common time stamp, averages and then plots everything on one plot i.e
# plots the average and also each flies trace in lighter colors. It does this for all the speeds
# i.e new plot per rotation speed. The trace for each fly represents its left - right baseline subtracted WBA.

########################

def resample(x1, y1, x2, kind, extrapolate=True):
	# helper function
	if kind == 'spline':
		spline = spi.CubicSpline(x1, y1)
		y2 = spline.__call__(x2, extrapolate=extrapolate)
	else:
		fill_value = "extrapolate" if extrapolate else []
		interp = spi.interp1d(x1, y1, kind=kind, bounds_error=False, fill_value=fill_value)
		y2 = interp(x2)
	return y2


def fourier_transform_to_get_cutoff(signal, t):

	### one alternative suggested by email to reduce noise was to take the second half of the signal,invert it as the first half and use this as the cleaned signal
	# N = signal.shape[0]
	# signal_second_half = signal[int(N/2)::]
	# ## move the signal down to 0 to fix asymmetry
	# signal_second_half = signal_second_half - signal_second_half[0]
	# signal_new = np.zeros_like(signal)
	# signal_new[0:(int(N%2) + int(N/2))] = -1.0*signal_second_half
	# signal_new[int(N/2)::] = signal_second_half

	# signal = signal_new



	# helper function for me to visualize and see 
	# fourier transform to freq domain and look at signal to see what cutoff value to use for low pass filter

	N = signal.shape[0]
	dt = t[2] - t[1]

	#print ('sampling freq:', 1.0/dt)

	fft = 1.0/N * np.fft.fft(signal)
	fft = fft[:N//2]
	fftfreq = np.fft.fftfreq(N, dt)[:N//2] 
	
	#plt.plot(fftfreq, np.abs(fft))
	#plt.show()
	
	# the first highest fft value is going to be a line ..look at fun with fft if you set the signal as 5 + sinx biggest spike is a line then is the one for the sine curve
	# so we can safely ignore the first fft value.. and we know its going to be at 0 freq i.e at fft[0]

	idx = 1 + np.argmax(np.abs(fft[1:]))
	#print ('index:', idx)

	amp = np.abs(fft[idx]) * 2
	#print ('amp:', amp)

	phase =  np.angle(fft[idx]) * 180/np.pi
	#print ('phase in degrees:',phase)

	freq = np.abs(fftfreq[idx])
	#print ('frequency:',freq)

	## add offset np.mean(signal) just for visualizing the fits better
	#phase = 90
	signal_reconstucted =  amp*np.cos(2*np.pi*freq*t + phase*np.pi/180) + np.mean(signal)


	### lets try doing an optimization here using scipy to increase fit quality
	optimize_func = lambda x: x[0]*np.cos(2*np.pi*x[1]*t+x[2]*np.pi/180) + np.mean(signal) - signal
	est_amp, est_freq, est_phase, est_mean = leastsq(optimize_func, [amp, freq, phase, np.mean(signal)])[0]
	#print ('##### scipy optimized #####')
	#print ('amp:', est_amp)
	#print ('phase', est_phase)
	#print ('frequency:', est_freq)
	optimized_signal_reconstructed =  est_amp*np.cos(2*np.pi*est_freq*t + est_phase*np.pi/180) + np.mean(signal)

	plt.clf()
	#plt.ylim(-8,8)
	return optimized_signal_reconstructed, est_amp, est_phase, signal

	
def calculate_average_over_cycles_per_fly(average_left_minus_right, t, speed, head, store_baselines=False, left=False, right=False):
	### calculate average over 5 sinusoid cycles per fly.. they are already resampled to have the same t
	list_of_amps_per_fly = []
	list_of_signals_per_fly = []
	list_of_signals_fitted_per_fly = []
	list_of_optimized_signals_fitted_per_fly = []
	list_of_vertical_offsets_per_fly = []

	num_cycles = 5
	t1 = int(round(t.shape[0]/float(num_cycles)))
	resampling_dt = np.mean(np.diff(t[0:t1]))
	t2 = np.arange(len(t[0:t1])) * resampling_dt
	plt.clf()
	

	for fly_number,WBA in enumerate(average_left_minus_right):
		### this loop is per fly
		temp = []
		for x in range(0, num_cycles):
			temp.append(resample(t[x*t1:(x+1)*t1] - t[x*t1] , WBA[x*t1:(x+1)*t1], t2, kind='linear'))
		average_over_sinusoid_cycles = np.mean(temp, axis=0)
		std_dev = np.std(temp, axis=0)
		# now we also want to fit a sine curve to this average_over_sinusoid_cycles 
		optimized_fitted_sin_curve, amp, phase, average_over_sinusoid_cycles  = fourier_transform_to_get_cutoff(average_over_sinusoid_cycles,t2)
		#list_of_amps_per_fly.append(amp)
		list_of_amps_per_fly.append(amp)
		
		list_of_signals_per_fly.append(average_over_sinusoid_cycles)
		##list_of_signals_fitted_per_fly.append(fitted_sin_curve)
		list_of_optimized_signals_fitted_per_fly.append(optimized_fitted_sin_curve)
		list_of_vertical_offsets_per_fly.append(np.mean(average_over_sinusoid_cycles))

	if show_plots:
		### now plot everything
		plt.figure(figsize=(5,5))
		
		for i in range(0,len(list_of_amps_per_fly)):
			plt.subplot(5,5,i+1)
			if head==0:
				plt.ylim(-40,40)
			if head==2:
				plt.ylim(-0.1,0.1)
			else:
				plt.ylim(-40,40)
			plt.title('amplitude of fit:' + str(round(list_of_amps_per_fly[i],2)) + ', R^2:')
			if i>=len(list_of_amps_per_fly)-3:
				plt.xlabel('Time (seconds)')
			if head==0:
				plt.ylabel('L-R WBA (degrees)')
			elif head==1:
				plt.ylabel('Head roll angle (degrees)')
			elif head==2:
				plt.ylabel('Head pitch (pixels)')
			elif head==3:
				plt.ylabel('Head yaw angle (degrees)')

			plt.plot(t2, list_of_signals_per_fly[i] , 'g', alpha=0.5)
			##plt.plot(t2, list_of_signals_fitted_per_fly[i], 'r', alpha=0.5)
			plt.plot(t2, list_of_optimized_signals_fitted_per_fly[i], 'r', alpha=0.5)
		
		
		##plt.title('fit for fly :' + str(1+fly_number) + ' amplitude :' + str(amp))
		##plt.title('all fits')
		plt.show()



	if head:
		## save to pickle file for plotting all genotypes on one plot
		if head==1:
			pickle.dump(list_of_amps_per_fly, open('/home/tarun/catkin_ws/src/trajectories_autostep_ros/src/data_for_scatter_plot/'+subdir[:-1]+'-speed-'+str(speed+1)+'-head-roll.p', 'wb'))
		elif head==2:
			pickle.dump(list_of_amps_per_fly, open('/home/tarun/catkin_ws/src/trajectories_autostep_ros/src/data_for_scatter_plot/'+subdir[:-1]+'-speed-'+str(speed+1)+'-head-pitch.p', 'wb'))
		elif head==3:
			pickle.dump(list_of_amps_per_fly, open('/home/tarun/catkin_ws/src/trajectories_autostep_ros/src/data_for_scatter_plot/'+subdir[:-1]+'-speed-'+str(speed+1)+'-head-yaw.p', 'wb'))

	else:
		## save to pickle file for plotting all genotypes on one plot
		if left:
			pickle.dump(list_of_amps_per_fly, open('/home/tarun/catkin_ws/src/trajectories_autostep_ros/src/data_for_scatter_plot/'+subdir[:-1]+'-left-speed-'+str(speed+1)+'.p', 'wb'))

		elif right:
			pickle.dump(list_of_amps_per_fly, open('/home/tarun/catkin_ws/src/trajectories_autostep_ros/src/data_for_scatter_plot/'+subdir[:-1]+'-right-speed-'+str(speed+1)+'.p', 'wb'))

		else:
			pickle.dump(list_of_amps_per_fly, open('/home/tarun/catkin_ws/src/trajectories_autostep_ros/src/data_for_scatter_plot/'+subdir[:-1]+'-speed-'+str(speed+1)+'.p', 'wb'))
			if store_baselines:
				pickle.dump([list_of_amps_per_fly, mean_baseline_left[str(speed)], mean_baseline_right[str(speed)]], open('/home/tarun/catkin_ws/src/trajectories_autostep_ros/src/data_for_baseline_correlation_plot/'+subdir[:-1]+'-speed-'+str(speed+1)+'.p', 'wb'))
				#pickle.dump([list_of_vertical_offsets_per_fly, mean_baseline_left[str(speed)], mean_baseline_right[str(speed)]], open('/home/tarun/catkin_ws/src/trajectories_autostep_ros/src/data_for_baseline_correlation_plot/'+subdir.split('-')[0]+'-'+subdir.split('-')[1]+'-speed-'+str(speed+1)+'-offsets.p', 'wb'))
				
			if show_plots:
				plt.scatter([0]*len(list_of_amps_per_fly),list_of_amps_per_fly)
				plt.ylim(0,30)
				plt.show()


## flag to indicate whether head tracking has been done already on this genotype or not
head_data = False
show_plots = True
subdir = 'UX28C05yawbothdirections/'
all_flies = glob.glob('/home/tarun/catkin_ws/src/trajectories_autostep_ros/bagfiles/hdf5/'+subdir+'/*')
pickle_files = []

for i in all_flies:
	pickle_files.append(glob.glob('/home/tarun/catkin_ws/src/trajectories_autostep_ros/bagfiles/hdf5/'+subdir+i.split('/')[-1]+'/avg_left_minus_right*.p'))
	

print (pickle_files)
##pickle_files = [['/home/tarun/catkin_ws/src/trajectories_autostep_ros/bagfiles/hdf5/UX-S00-top-view/Nov-2-fly5/avg_left_minus_right_Nov-2-fly5.p'], ['/home/tarun/catkin_ws/src/trajectories_autostep_ros/bagfiles/hdf5/UX-S00-top-view/Nov-2-fly7/avg_left_minus_right_Nov-2-fly7.p'], ['/home/tarun/catkin_ws/src/trajectories_autostep_ros/bagfiles/hdf5/UX-S00-top-view/Nov-1-fly1/avg_left_minus_right_Nov-1-fly1.p'], ['/home/tarun/catkin_ws/src/trajectories_autostep_ros/bagfiles/hdf5/UX-S00-top-view/Nov-1-fly2/avg_left_minus_right_Nov-1-fly2.p'], ['/home/tarun/catkin_ws/src/trajectories_autostep_ros/bagfiles/hdf5/UX-S00-top-view/Nov-2-fly6/avg_left_minus_right_Nov-2-fly6.p'], ['/home/tarun/catkin_ws/src/trajectories_autostep_ros/bagfiles/hdf5/UX-S00-top-view/Nov-2-fly4/avg_left_minus_right_Nov-2-fly4.p'], ['/home/tarun/catkin_ws/src/trajectories_autostep_ros/bagfiles/hdf5/UX-S00-top-view/Nov-2-fly3/avg_left_minus_right_Nov-2-fly3.p']]

num_speeds = 2
trajectories = []
trajectories_time = []

mean_baseline_left = {}
mean_baseline_right = {}


for i in range(0, num_speeds):
	mean_baseline_left[str(i)] = []
	mean_baseline_right[str(i)] = []

	# new plot for each speed
	if show_plots:
		figures,axes = plt.subplots()

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
		WBA = dict_for_pickle[i][1]
		head_yaw = dict_for_pickle[i][2]

		left_WBA = dict_for_pickle['mean_left_'+str(i)]
		right_WBA = dict_for_pickle['mean_right_'+str(i)]


		if head_data:
			head_roll = dict_for_pickle[i][2]
			head_pitch = dict_for_pickle[i][3]
			head_yaw = dict_for_pickle[i][4]
		
		if 'trajectory_1' and 'trajectory_1_time' in dict_for_pickle.keys():
			for x in range(0,num_speeds):
				trajectories.append(dict_for_pickle['trajectory_'+str(x)][0])
				trajectories_time.append(dict_for_pickle['trajectory_'+str(x)+'_time'][0])

		if 'mean_baseline_left_1' in dict_for_pickle.keys():
			mean_baseline_left[str(i)].append(dict_for_pickle['mean_baseline_left_'+str(i)])
			mean_baseline_right[str(i)].append(dict_for_pickle['mean_baseline_right_'+str(i)])


		if head_data:
			time_and_WBA_for_all_flies.append([time, WBA, left_WBA, right_WBA, head_roll, head_pitch, head_yaw])
		else:
			#if abs(dict_for_pickle['mean_baseline_left_'+str(i)] - dict_for_pickle['mean_baseline_right_'+str(i)]) < 10:
			time_and_WBA_for_all_flies.append([time, WBA, left_WBA, right_WBA, head_yaw])

		# first lets just plot all of them on one plot
		
		#axes.set_ylim(-40,40)
		if show_plots:
			r = random.random()
			b = random.random()
			g = random.random()
			color = (r, g, b)
			axes.set_xlabel('Time (seconds)')
			axes.set_ylabel('L-R WBA (degrees)')

			axes.plot(time, WBA, c=color, alpha=0.5)


	# lets just resample based on the time of the first one in the list 

	resampling_dt = np.mean(np.diff(time_and_WBA_for_all_flies[0][0]))

	t = np.arange(len(time_and_WBA_for_all_flies[0][0])) * resampling_dt
	average_left_minus_right = np.zeros((len(time_and_WBA_for_all_flies),t.shape[0]))
	average_head_roll = np.zeros_like(average_left_minus_right)
	average_head_pitch = np.zeros_like(average_left_minus_right)
	average_head_yaw = np.zeros_like(average_left_minus_right)
	average_left = np.zeros_like(average_left_minus_right)
	average_right = np.zeros_like(average_left_minus_right)


	for k,fly in enumerate(time_and_WBA_for_all_flies):
		WBA = resample(fly[0], fly[1], t, kind='linear')
		left_WBA = resample(fly[0], fly[2], t, kind='linear')
		right_WBA = resample(fly[0], fly[3], t, kind='linear')
		yaw = resample(fly[0], fly[4], t, kind='linear')

		### mean subtract every trace so that they all start roughly at 0 offset on the Y axis
		#average_left_minus_right[k] = WBA - np.mean(WBA)
		average_left_minus_right[k] = WBA
		average_left[k] = left_WBA
		average_right[k] = right_WBA
		average_head_yaw[k] = yaw

		if head_data:
			### mean subtract every trace so that they all start roughly at 0 offset on the Y axis
			roll = resample(fly[0], fly[4], t, kind='linear')
			pitch = resample(fly[0], fly[5], t, kind='linear')
			yaw = resample(fly[0], fly[6], t, kind='linear')
			
			average_head_roll[k] = roll
			average_head_pitch[k] = pitch
			average_head_yaw[k] = yaw
		
	std = np.std(average_left_minus_right, axis = 0)
	mean = np.mean(average_left_minus_right, axis = 0)
	mean_left = np.mean(average_left, axis=0)
	mean_right = np.mean(average_right, axis=0)
	mean_yaw = np.mean(average_head_yaw, axis = 0)


	if head_data:
		mean_roll = np.mean(average_head_roll, axis = 0)
		mean_pitch = np.mean(average_head_pitch, axis = 0)
		mean_yaw = np.mean(average_head_yaw, axis = 0)

	
	if show_plots:
		axes.plot(t, mean, c='r', label='left - right (avg)')
		axes.set_ylim(-40,40)
		axes.set_xlim(0, t[-1])
		axes.set_ylabel('L - R WBA (red)')
	
	# have to resample even the rockafly trajectory
	resampled_trajectory = resample(trajectories_time[i], trajectories[i], t, kind='linear')
	if show_plots:
		axes3 = axes.twinx()
		axes3.plot(t, resampled_trajectory,c='b')
		#axes3.set_yticklabels([])
		axes3.set_ylabel('Angular velocity of spinning arm (blue)')
		plt.title(subdir + ' n = '+str(average_left_minus_right.shape[0]) + ' speed ' + str(i+1))
		plt.xlabel('Time (seconds)')
		#plt.ylabel('R WBA (degrees)')

		plt.show()
		plt.clf()		
		plt.ylim(-40,40)
		plt.xlabel('Time (seconds)')
		plt.ylabel('Head yaw angle (degrees) ')
		plt.plot(t, mean_yaw, c='r', label='yaw')
		#plt.plot(t, resampled_trajectory, c='b')
		plt.show()

		###############		
		if head_data:
			plt.clf()		
			plt.ylim(-30,30)
			plt.xlabel('Time (seconds)')
			plt.ylabel('Head roll angle (degrees) ')
			plt.plot(t, mean_roll, c='r', label='left - right (avg)')
			plt.show()





	## we want, for each fly, the average over 5 sinusoid cycles and the value of the fit to that in order to plot on a scatter plot
	## the average_left_minus_right containts WBA for all flies for a particular speed, resampled already to a common 't'
	calculate_average_over_cycles_per_fly(average_left_minus_right, t, i, head=0, store_baselines=True)

	#calculate_average_over_cycles_per_fly(average_left, t, i, head=0, store_baselines=False, left=True)
	#calculate_average_over_cycles_per_fly(average_right, t, i, head=0, store_baselines=False, left=False, right=True)

	calculate_average_over_cycles_per_fly(average_head_yaw, t, i, head=3)



	## do the same for the head
	if head_data:
		calculate_average_over_cycles_per_fly(average_head_roll, t, i, head=1)
		calculate_average_over_cycles_per_fly(average_head_pitch, t, i, head=2)
		calculate_average_over_cycles_per_fly(average_head_yaw, t, i, head=3)


	#### also plot the average over the 5 sinusoid cycles of the mean trace (over all flies)
	num_cycles = 5
	

	temp = []
	temp_head = []
	t1 = int(round(t.shape[0]/float(num_cycles)))
	
	# we want to resample the 5 sinusoids to length t1 because the total number of indexes might not be divisible by 5.

	# for z in range(0,average_left_minus_right.shape[0]):
	# 	plt.plot(range(0,mean.shape[0]), average_left_minus_right[z,:])
	# plt.show()


	resampling_dt = np.mean(np.diff(t[0:t1]))
	t2 = np.arange(len(t[0:t1])) * resampling_dt

	for x in range(0, num_cycles):
		temp.append(resample(t[x*t1:(x+1)*t1] - t[x*t1] , mean[x*t1:(x+1)*t1], t2, kind='linear'))

		temp_head.append(resample(t[x*t1:(x+1)*t1] - t[x*t1] , mean_yaw[x*t1:(x+1)*t1], t2, kind='linear'))
		
		if head_data:
			temp_head.append(resample(t[x*t1:(x+1)*t1] - t[x*t1] , mean_roll[x*t1:(x+1)*t1], t2, kind='linear'))

	average_over_sinusoid_cycles = np.mean(temp, axis=0)
	std_dev = np.std(temp, axis=0)

	average_over_sinusoid_cycles_yaw = np.mean(temp_head, axis=0)
	std_dev_yaw = np.std(temp_head, axis=0)

	if head_data:
		average_over_sinusoid_cycles_roll = np.mean(temp_head, axis=0)	
		std_dev_roll = np.std(temp_head, axis=0)

	#### now we also want to fit a sine curve to this average_over_sinusoid_cycles 
	#optimized_fitted_sin_curve, amp, phase, _ = fourier_transform_to_get_cutoff(average_over_sinusoid_cycles,t2)
	
	#if show_plots:
	plt.clf()
	figure, axes = plt.subplots()
	## plot everything
	
	axes.plot(t[0:t1], average_over_sinusoid_cycles, 'g')
	axes.set_xlabel('Time (seconds)')
	axes.set_ylabel(' L - R WBA (degrees)')
	axes.set_ylim(-40,40)
	#plt.ylim(-25,25)
	##axes.plot(t[0:t1], fitted_sin_curve, 'r')
	
	#plt.fill_between(t[0:t1], average_over_sinusoid_cycles-std_dev, average_over_sinusoid_cycles+std_dev,color='green',alpha=0.2)
	axes2 = axes.twinx()
	axes2.plot(t[0:t1], resampled_trajectory[0:t1], 'b')
	axes2.set_ylabel('Stimulus angular velocity (degrees/second)')
	#plt.xlabel('Time (seconds)')
	#plt.ylabel('L - R WBA (degrees)')
	plt.title(subdir + ' n = '+str(average_left_minus_right.shape[0]) + ' avg over ' + str(num_cycles) + ' cycles' + ' speed ' + str(i+1) )
	plt.show()
	plt.clf()


	#optimized_fitted_sin_curve, amp, phase, _ = fourier_transform_to_get_cutoff(average_over_sinusoid_cycles_yaw,t[0:t1])
	## plot everything
	figure, axes = plt.subplots()
	# plot everything
	axes.plot(t[0:t1], average_over_sinusoid_cycles_yaw, 'g')
	axes.set_xlabel('Time (seconds)')
	axes.set_ylabel(' Head yaw (degrees)')
	axes.set_ylim(-20,20)
	#plt.ylim(-25,25)
	##axes.plot(t[0:t1], fitted_sin_curve, 'r')
	
	#plt.fill_between(t[0:t1], average_over_sinusoid_cycles-std_dev, average_over_sinusoid_cycles+std_dev,color='green',alpha=0.2)
	axes2 = axes.twinx()
	axes2.plot(t[0:t1], resampled_trajectory[0:t1], 'b')
	axes2.set_ylabel('Stimulus angular velocity (degrees/second)')
	#plt.xlabel('Time (seconds)')
	#plt.ylabel('L - R WBA (degrees)')
	plt.title(subdir + ' head yaw n = '+str(average_left_minus_right.shape[0]) + ' avg over ' + str(num_cycles) + ' cycles' + ' speed ' + str(i+1) )
	plt.show()
	plt.clf()


	if head_data:
		#optimized_fitted_sin_curve, amp, phase, _ = fourier_transform_to_get_cutoff(average_over_sinusoid_cycles_roll,t[0:t1])
		## plot everything
		plt.plot(t[0:t1], average_over_sinusoid_cycles_roll, 'g')
		plt.fill_between(t[0:t1], average_over_sinusoid_cycles_roll-std_dev_roll, average_over_sinusoid_cycles_roll+std_dev_roll,color='green',alpha=0.2)
		plt.ylim(-25,25)
		plt.plot(t[0:t1], optimized_fitted_sin_curve, 'r')			
		plt.xlabel('Time (seconds)')
		plt.ylabel('Head roll angle (degrees)')
		plt.title('head roll ' + subdir + ' n = '+str(average_left_minus_right.shape[0]) + ' avg over ' + str(num_cycles) + ' cycles' + ' speed ' + str(i+1) )
		plt.show()
		plt.clf()


	########### make nice seaborn plots which automatically gives confidence intervals ##############################
	
	num_cycles = 5
	temp = []
	temp_head = []
	t1 = int(round(t.shape[0]/float(num_cycles)))
	
	# we want to resample the 5 sinusoids to length t1 because the total number of indexes might not be divisible by 5.
	resampling_dt = np.mean(np.diff(t[0:t1]))
	t2 = np.arange(len(t[0:t1])) * resampling_dt

	for f in range(0, len(average_left_minus_right)):
		f_WBA = average_left_minus_right[f]
		f_head_yaw = average_head_yaw[f]

		for x in range(0, num_cycles):
			temp.append(resample(t[x*t1:(x+1)*t1] - t[x*t1] , f_WBA[x*t1:(x+1)*t1], t2, kind='linear'))

			temp_head.append(resample(t[x*t1:(x+1)*t1] - t[x*t1] , f_head_yaw[x*t1:(x+1)*t1], t2, kind='linear'))
			
	
	# # ########################### I want to this plot where I have cycle wise collapsed traces in order to measure drift ########
	# figure, axes = plt.subplots()
	# cmap = plt.get_cmap('jet', 5)
	
	# for outer in range(0,5):
	# 	cycle_list = []
	# 	for f in range(outer, len(temp_head), 5):
	# 		cycle_list.append(temp_head[f])

	# 	sns.lineplot(list(t2)*len(cycle_list), list(np.array(cycle_list).flatten()),ax = axes, color=cmap(outer), alpha=1-(0.1*outer))
	# 	axes.set_xlim([0,t2[-1]])
	# 	axes.set_ylim([-20,20])
	# 	axes.set_xlabel('Time (seconds)')
	# 	axes.set_ylabel('Head yaw  (degrees)')

	# norm = mpl.colors.Normalize(vmin=0, vmax=5)
	  
	# # creating ScalarMappable
	# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
	# sm.set_array([])
	  
	# plt.colorbar(sm, ticks=np.linspace(1, 5, 5))

	# plt.show()
	# plt.clf()



	##############################################################################################################################



	figure, axes = plt.subplots()
	
	sns.lineplot(list(t2)*len(temp), list(np.array(temp).flatten()),ax = axes, color='r')
	axes.set_xlim([0,t2[-1]])
	axes.set_ylim([-40,40])
	axes.set_xlabel('Time (seconds)')
	axes.set_ylabel('WBA yaw (degrees)')

	axes2 = axes.twinx()
	axes2.plot(t2, resampled_trajectory[0:t1], 'b' )
	axes2.set_ylabel('Stimulus angular velocity (degrees/second)')
	plt.title(subdir + ' WBA yaw n = '+str(average_left_minus_right.shape[0]) + ' avg over 5 cycles ' + ' speed ' + str(i+1))


	plt.show()