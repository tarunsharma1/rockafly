#### this is part of making overlay videos, i.e overlay of the fly camera_stream video with a small moving plot of kinefly L-R value (top right). Later we are 
#### also going to overlay headtracker roll value (top left). To do this I am going to save kinefly l-r value as a sequence of images (remember already there is only 1 kinefly value per frame).
#### Then I am going to open each frame of the actual fly video, and crop in the image of the saved plot for that frame in the top right, and then save the whole new frame as a video.
#### To do the kinefly part, I will need the kinefly values from the hdf5 file and also the a3 values similar to headtracking.

import glob
import os
import copy
import pathlib2 as pathlib
import numpy as np
import cv2
import scipy.fftpack
import scipy.signal as sps
import matplotlib.pyplot as plt
import pickle
import sys
#sys.path.append("../")
from rkf_analysis_tarun import RkfAnalysis


class OverlayVideo():
	def __init__(self, hdf5_path, trial_predictions_path, video_path):
		self.number_of_speeds = 3
		self.hdf5_path = hdf5_path
		self.trial_predictions_path = trial_predictions_path
		self.video_path = video_path

		# fill these in later
		self.headtracking_values = {}
		self.headtracking_values['yaw'] = []
		self.headtracking_values['pitch'] = []
		self.headtracking_values['roll'] = []
		

		# get segregation indices, and trajectory information
		self.fly = RkfAnalysis(self.number_of_speeds)
		self.fly.read_hdf5(self.hdf5_path)
		self.fly.find_trajectory_speed_change_points()
		self.fly.segregate_based_on_command_timings()
		self.fly.nan_interp()
		# because we will be plotting the trajectory too, we call the following function to make sure
		# the timestamps of the trajectory and kinefly timing (in this case also used as headtracking timing)
		# are of the same ranges etc.
		self.fly.resample_kinefly_and_autostep_data()


	def make_headtracking_plots(self):
		# TODO : add butterworth lpf to headtracking values before plotting
		# make headtracking per frame plot
		# values are already there in the predictions file, just need to plot it same as kinefly
		file = open(self.trial_predictions_path + '/alexnet_predictions.txt', 'r')
		ypr_values = file.readlines()

		pitch_values = []
		roll_values = []
		yaw_values = []

		for k in range(0,len(ypr_values)):
			# remove \n
			l = ypr_values[k][:-1]
			pitch_values.append(float(l.split(',')[1]))
			roll_values.append(float(l.split(',')[2]))
			yaw_values.append(float(l.split(',')[3]))


		pitch_values = np.array(pitch_values)
		roll_values = np.array(roll_values)
		yaw_values = np.array(yaw_values)

		# low pass filter them
		pitch_values, roll_values, yaw_values = self.butterworth_filter(pitch_values, roll_values, yaw_values)

		total_number_of_frames = yaw_values.shape[0]
		frames = np.arange(0,total_number_of_frames)

		final_headtracking_plots = np.zeros((total_number_of_frames,200,200,1))

		fig, axes = plt.subplots()
		
		
		for i in range(0, total_number_of_frames):
			axes.set_ylim(-30,30)
			
			if i>400:
				axes.plot(frames[i-400:i], roll_values[i-400:i])
				#axes2.plot(frames[i-400:i], trajectory[i-400:i])
			
				fig.canvas.draw()
				img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
				img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
				img = cv2.resize(img,(200,200), interpolation = cv2.INTER_AREA)
				# img is rgb, convert to opencv's default bgr
				img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
				
				final_headtracking_plots[i] = np.expand_dims(img,axis=-1)
				#cv2.imshow('plot', img)
				#cv2.waitKey(1)
				axes.clear()
				
		return final_headtracking_plots


	def butterworth_filter(self, pitch_values, roll_values, yaw_values):
		# lets smooth the shit out of this
		order=5.0
		cutoff=3
	
		# this is after resampling so should be constant
		#kf_dt = self.fly.kinefly_data_separated_on_speed['timestamp'][i][2] - self.fly.kinefly_data_separated_on_speed['timestamp'][i][1] 
		nyquist = 10.0
		#print (nyquist)
		bfilt = sps.butter(order, cutoff/nyquist, btype='lowpass')
		yaw_values = sps.filtfilt(bfilt[0], bfilt[1], yaw_values)
		roll_values = sps.filtfilt(bfilt[0], bfilt[1], roll_values)
		pitch_values = sps.filtfilt(bfilt[0], bfilt[1], pitch_values)

		return pitch_values, roll_values, yaw_values

		

	def make_kinefly_per_frame_plots(self):
		# make the kinefly per frame plot
		# i need to get the kinefly values from the hdf5 for the baseline periods, the segregation points and also the resampled values for the spinning period
		total_number_of_frames = self.fly.kinefly_data['left'].shape[0]
		frames = np.arange(0,total_number_of_frames)

		final_kinefly_plots = np.zeros((total_number_of_frames,200,200,1))

		# each value in this list corresponds to the the kinefly value for that frame (i.e index is the frame)
		main_list_left = []
		main_list_right = []

		# initial baseline
		main_list_left.append(self.fly.kinefly_data['left'][0:self.fly.a3[0][0]])  
		main_list_right.append(self.fly.kinefly_data['right'][0:self.fly.a3[0][0]])  
		
		# first rotation (these are resampled values but don't worry because their number is still the same they are just uniformly spaced now)
		# main_list_left.append(self.fly.kinefly_data_separated_on_speed['left'][0])
		# main_list_right.append(self.fly.kinefly_data_separated_on_speed['right'][0])
		
		# # second baseline
		# main_list_left.append(self.fly.kinefly_data['left'][self.fly.a3[0][0] + self.fly.kinefly_data_separated_on_speed['left'][0].shape[0]:self.fly.a3[1][0]])
		# main_list_right.append(self.fly.kinefly_data['right'][self.fly.a3[0][0] + self.fly.kinefly_data_separated_on_speed['right'][0].shape[0]:self.fly.a3[1][0]])
		
		# # second rotation
		# main_list_left.append(self.fly.kinefly_data_separated_on_speed['left'][1])
		# main_list_right.append(self.fly.kinefly_data_separated_on_speed['right'][1])
		
		# # final baseline
		# main_list_left.append(self.fly.kinefly_data['left'][self.fly.a3[1][0] + self.fly.kinefly_data_separated_on_speed['left'][1].shape[0]::])
		# main_list_right.append(self.fly.kinefly_data['right'][self.fly.a3[1][0] + self.fly.kinefly_data_separated_on_speed['right'][1].shape[0]::])

		for i in range(0, self.number_of_speeds-1):
			# during spin
			main_list_left.append(self.fly.kinefly_data_separated_on_speed['left'][i])
			main_list_right.append(self.fly.kinefly_data_separated_on_speed['right'][i])

			# between spins
			main_list_left.append(self.fly.kinefly_data['left'][self.fly.a3[i][0] + self.fly.kinefly_data_separated_on_speed['left'][i].shape[0]:self.fly.a3[i+1][0]])
			main_list_right.append(self.fly.kinefly_data['right'][self.fly.a3[i][0] + self.fly.kinefly_data_separated_on_speed['right'][i].shape[0]:self.fly.a3[i+1][0]])

		# final spin
		main_list_left.append(self.fly.kinefly_data_separated_on_speed['left'][i])
		main_list_right.append(self.fly.kinefly_data_separated_on_speed['right'][i])

		# final baseline 
		main_list_left.append(self.fly.kinefly_data['left'][self.fly.a3[i][0] + self.fly.kinefly_data_separated_on_speed['left'][i].shape[0]::])
		main_list_right.append(self.fly.kinefly_data['right'][self.fly.a3[i][0] + self.fly.kinefly_data_separated_on_speed['right'][i].shape[0]::])


		# merge them all into one list
		main_list_left = [item for sublist in main_list_left for item in sublist]
		main_list_right = [item for sublist in main_list_right for item in sublist]
		
		left_minus_right = np.array(main_list_left) - np.array(main_list_right)
		
		# we also want the trajectory info per frame
		trajectory = np.zeros((total_number_of_frames))
		
		# add the rotations data, the rest remains 0s
		for i in range(0,self.number_of_speeds):
			trajectory[self.fly.a3[i]] = self.fly.trajectory_data_separated_on_speed['position'][i]
		

		# now we want to make, for each frame, a plot showing the kinefly l-r value for that frame but also of the previous lets say 50 frames to achieve moving animataion effect
		# first 400 frames get no plot after that there is an image to be returned for every frame
		fig, axes = plt.subplots()
		#axes.set_ylim(-60,60)
		axes2 = axes.twinx()

		
		for i in range(0, total_number_of_frames-1):
			axes.set_ylim(-90,90)
			axes2.set_ylim(-2000,2000)	
			
			if i>400:
				axes.plot(frames[i-400:i], left_minus_right[i-400:i])
				axes2.plot(frames[i-400:i], trajectory[i-400:i])
			
				fig.canvas.draw()
				img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
				img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
				img = cv2.resize(img,(200,200), interpolation = cv2.INTER_AREA)
				# img is rgb, convert to opencv's default bgr
				img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
				
				final_kinefly_plots[i] = np.expand_dims(img,axis=-1)
				#cv2.imshow('plot', img)
				#cv2.waitKey(1)
				axes.clear()
				axes2.clear()

		return final_kinefly_plots


	def open_video_and_write(self, final_kinefly_plots, final_headtracking_plots):
		location_to_write = '/'.join(self.video_path.split('/')[:-1]) + '/'

		if not os.path.exists(location_to_write + 'overlaid_videos/'):
			os.makedirs(location_to_write + 'overlaid_videos/')
		
		out = cv2.VideoWriter(location_to_write + 'overlaid_videos/'+ self.video_path.split('/')[-1], cv2.VideoWriter_fourcc('M','J','P','G'), 30, (600,600))

		cap = cv2.VideoCapture(self.video_path)
		print self.video_path
		frame_number = 0
		while(cap.isOpened()):
			# Capture frame-by-frame
			ret, frame = cap.read()
			
			if ret == True:
				# crop in the correponding plot
				frame[0:200,0:200,:] = final_kinefly_plots[frame_number]
				frame[0:200:,400:600,:] = final_headtracking_plots[frame_number]

				# Display the resulting frame
				#cv2.imshow('Frame',frame)
				#cv2.waitKey(30)
				out.write(frame)
				frame_number += 1
			else:
				break
		cap.release()
		



if __name__ == '__main__':
	## did UX-S333/Nov-20-fly5 but redo cause missing trajectory for slow speed
	## also run DAKir-S333/nov12-fly3
	fly_path = '/hdd/catkin_ws/src/trajectories_autostep_ros/bagfiles/'
	fly_name = 'Jan-18-fly2/'
	hdf5 = 'hdf5/roll/UX-S00-redo/' + fly_name
	headtracking_predictions = 'camera_streams/roll/UX-S00-redo/headtracking/' + fly_name
	video_path = fly_path + 'camera_streams/roll/UX-S00-redo/' + fly_name

	trial_paths = glob.glob(fly_path + hdf5 + '/*.hdf5') # use glob
	
	print ('found ' + str(len(trial_paths)) + ' trials')

	# lets just do one trial for now to save time
	trial_paths = [trial_paths[2]]
	print (trial_paths)
	import ipdb;ipdb.set_trace()

	# a list of objects one for each trial
	trial_objects = []
	for i,trial_hdf5 in enumerate(trial_paths):
		#trial_hdf5 = fly_path + hdf5 + trial_pred.split('/')[-1]
		trial_pred = fly_path + headtracking_predictions + trial_hdf5.split('/')[-1][:-16] + '/'
		T = OverlayVideo(trial_hdf5, trial_pred, video_path + trial_hdf5.split('/')[-1][:-16] + '.avi')
		final_headtracking_plots = T.make_headtracking_plots()
		print ('done headtracking plots')
		
		final_kinefly_plots = T.make_kinefly_per_frame_plots()
		print ('done kinefly plots')
		T.open_video_and_write(final_kinefly_plots, final_headtracking_plots)
		

		# these come later for headtracking values
		#T.read_predictions_and_segregate()
		#T.butterworth_filter()


