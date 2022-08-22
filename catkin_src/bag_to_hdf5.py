#! /usr/bin/python2



##### 

# written by : tarun
# notes : this file converts the bag files to hdf5 for easier access
#         it also converts angles from radians to degrees and inserts nans when no value present

#########


import rosbag
import numpy as np
import os
import cv2
import sys
from matplotlib import pyplot as plt
import h5py
import ipdb
import glob
#from tqdm import tqdm

for main_dir in ['UX28C05yawneg/', 'UX28C05yawpos/']: 

	#main_dir = 'UXJ79yawredoneg/'
	master_dir = '/home/tarun/catkin_ws/src/trajectories_autostep_ros/bagfiles/'

	sub_dirs = glob.glob(master_dir + main_dir + '*')

	## manual override ###
	#sub_dirs = ['/home/tarun/catkin_ws/src/trajectories_autostep_ros/bagfiles/UXJ90yawredoneg/Jun-14-fly3']

	#sub_dirs = ['/home/tarun/catkin_ws/src/trajectories_autostep_ros/bagfiles/UXJ90yawredopos/Jun-16-fly12', '/home/tarun/catkin_ws/src/trajectories_autostep_ros/bagfiles/UXJ90yawredopos/Jun-16-fly14', '/home/tarun/catkin_ws/src/trajectories_autostep_ros/bagfiles/UXJ90yawredopos/Jun-16-fly16', '/home/tarun/catkin_ws/src/trajectories_autostep_ros/bagfiles/UXJ90yawredopos/Jun-16-fly18', '/home/tarun/catkin_ws/src/trajectories_autostep_ros/bagfiles/UXJ90yawredopos/Jun-16-fly19']
	#####################

	for s_dir in sub_dirs:
		print ('############ NOW UNBAGGING '+s_dir + ' ###############')
		sub_directory = main_dir + s_dir.split('/')[-1] + '/'
		bag_files = glob.glob(master_dir + sub_directory + '*.bag')
		names = []
		for i in bag_files:
			names.append(i.split('/')[-1])
		print ('found '+str(len(names))+' bagfiles')

		for name in names:

			bag_file_name = master_dir + sub_directory + name
			print ('bag file:'+ bag_file_name)
			inbag = rosbag.Bag(bag_file_name)
			topics = inbag.get_type_and_topic_info()[1].keys()
			print (topics)
			#import ipdb;ipdb.set_trace()
			print ('getting sizes')
			kinefly_camera_data_size = inbag.get_message_count('/stabilized_image')
			#top_camera_data_size = inbag.get_message_count('/camera_uvc_resized_image')
			kinefly_data_size = inbag.get_message_count('/kinefly/flystate')
			trajectory_data_size = inbag.get_message_count('/autostep/motion_data')
			command_data_size = inbag.get_message_count('/sinusoid_trajectory_info')

			IMU_time_data_size = inbag.get_message_count('/IMU_time')
			IMU_acc_data_size = inbag.get_message_count('/IMU_acc_data')
			IMU_gyro_data_size = inbag.get_message_count('/IMU_gyro_data')


			#print ('number of frames :', top_camera_data_size)


			# # have to resample data
			# # kinefly_data[0-29500][1].head
			# # kinefly_data[0-29500][1].head.angles[0] gives angle
			# # kinefly_data[0][1].left.angles[0] * 180 / np.pi
			# # kinefly_data[0][1].header.stamp.to_sec()


			# make directory for the hdf5
			if not os.path.exists(master_dir + 'hdf5/'+ sub_directory):
				os.makedirs(master_dir + 'hdf5/' + sub_directory)


			hf = h5py.File(master_dir + 'hdf5/' + sub_directory + name.split('.bag')[0]+'_compressed.hdf5', 'w')

			#hf.create_dataset('frames',shape = (kinefly_camera_data_size,) + (600,600), compression='lzf')
			#hf.create_dataset('top_camera_frames_timestamp',shape = (top_camera_data_size,),dtype = 'float64', compression='lzf')
			hf.create_dataset('kinefly_camera_frames_timestamp',shape = (kinefly_camera_data_size,),dtype = 'float64', compression='lzf')
			
			hf.create_dataset('kinefly_data_head',shape = (kinefly_data_size,), dtype = 'float64', compression='lzf') # not sure what to do if value is missing..is nan the way to go?
			hf.create_dataset('kinefly_data_left',shape = (kinefly_data_size,), dtype = 'float64', compression='lzf')
			hf.create_dataset('kinefly_data_right',shape = (kinefly_data_size,), dtype = 'float64', compression='lzf')
			hf.create_dataset('kinefly_data_aux',shape = (kinefly_data_size,), dtype = 'float64', compression='lzf')
			hf.create_dataset('kinefly_data_timestamp',shape = (kinefly_data_size,), dtype = 'float64', compression='lzf')

			hf.create_dataset('trajectory_data',shape = (trajectory_data_size,), dtype = 'float64', compression='lzf')
			hf.create_dataset('trajectory_data_timestamp',shape = (trajectory_data_size,), dtype = 'float64', compression='lzf')

			hf.create_dataset('command_data',shape = (command_data_size,), dtype = 'float64', compression='lzf')
			hf.create_dataset('command_data_timestamp',shape = (command_data_size,), dtype = 'float64', compression='lzf')

			hf.create_dataset('IMU_time',shape = (IMU_time_data_size,),dtype='float64',compression='lzf')

			hf.create_dataset('IMU_gyro_x_data',shape = (IMU_gyro_data_size,),dtype='float64',compression='lzf')
			hf.create_dataset('IMU_gyro_y_data',shape = (IMU_gyro_data_size,),dtype='float64',compression='lzf')
			hf.create_dataset('IMU_gyro_z_data',shape = (IMU_gyro_data_size,),dtype='float64',compression='lzf')
			hf.create_dataset('IMU_acc_x_data',shape = (IMU_acc_data_size,),dtype='float64',compression='lzf')
			hf.create_dataset('IMU_acc_y_data',shape = (IMU_acc_data_size,),dtype='float64',compression='lzf')
			hf.create_dataset('IMU_acc_z_data',shape = (IMU_acc_data_size,),dtype='float64',compression='lzf')
			

			## write to video file
			frame_width,frame_height = 500,500

			# make directory for video
			if not os.path.exists(master_dir + 'camera_streams/' + sub_directory):
				os.makedirs(master_dir + 'camera_streams/' + sub_directory)

			out = cv2.VideoWriter(master_dir + 'camera_streams/' + sub_directory + name.split('.bag')[0]+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

			#out_top = cv2.VideoWriter(master_dir + 'camera_streams/' + sub_directory + name.split('.bag')[0]+'_top.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (450,340))

			

			rad2deg = 180 / np.pi
			kinefly_camera_data_idx = 0
			top_camera_data_idx = 0
			kinefly_data_idx = 0
			trajectory_data_idx = 0
			command_data_idx = 0
			IMU_time_idx = 0
			IMU_acc_data_idx = 0
			IMU_gyro_data_idx = 0

			for topic, msg, t in inbag.read_messages():
				
				if topic == '/stabilized_image':
					img_np_arr = np.frombuffer(msg.data, np.uint8)
					
					img_np_arr = img_np_arr.reshape((frame_width,frame_height))
					# hf['frames'][camera_data_idx] = img_np_arr
					#hf['frames_timestamp'][camera_data_idx] = t.to_time()
					#camera_data_idx += 1

					## write as a video file
					img_np_arr= np.expand_dims(img_np_arr, axis=-1)
					img_np_arr = cv2.cvtColor(img_np_arr, cv2.COLOR_GRAY2BGR)
					out.write(img_np_arr)

					hf['kinefly_camera_frames_timestamp'][kinefly_camera_data_idx] = t.to_time()				
					kinefly_camera_data_idx += 1


				# if topic == '/camera_uvc_resized_image':
				# 	img_np_arr = np.frombuffer(msg.data, np.uint8)
				# 	#import ipdb;ipdb.set_trace()
				# 	img_np_arr = img_np_arr.reshape((340,450,1))
				# 	img_np_arr = cv2.cvtColor(img_np_arr, cv2.COLOR_GRAY2BGR)

				# 	# hf['frames'][camera_data_idx] = img_np_arr
				# 	#hf['frames_timestamp'][camera_data_idx] = t.to_time()
				# 	#camera_data_idx += 1

				# 	## write as a video file
				# 	#import ipdb;ipdb.set_trace()
				# 	#img_np_arr = cv2.cvtColor(img_np_arr, cv2.COLOR_BGR2GRAY)
									
				# 	#img_np_arr= np.expand_dims(img_np_arr, axis=-1)
				# 	#import ipdb;ipdb.set_trace()
					
				# 	out_top.write(img_np_arr)

				# 	hf['top_camera_frames_timestamp'][top_camera_data_idx] = t.to_time()
				# 	top_camera_data_idx += 1



				if topic == '/kinefly/flystate':
					#import ipdb;ipdb.set_trace()
					hf['kinefly_data_head'][kinefly_data_idx] = msg.head.angles[0]*rad2deg if msg.head.angles else np.nan
					hf['kinefly_data_left'][kinefly_data_idx] = msg.left.angles[0]*rad2deg if msg.left.angles else np.nan
					hf['kinefly_data_right'][kinefly_data_idx] = msg.right.angles[0]*rad2deg if msg.right.angles else np.nan
					hf['kinefly_data_aux'][kinefly_data_idx] = msg.aux.angles[0]*rad2deg if msg.aux.angles else np.nan
					hf['kinefly_data_timestamp'][kinefly_data_idx] = t.to_time()
					#if kinefly_data_idx == 2500:
					#	import ipdb;ipdb.set_trace()
					kinefly_data_idx += 1

				if topic == '/autostep/motion_data':
					hf['trajectory_data'][trajectory_data_idx] = msg.position
					hf['trajectory_data_timestamp'][trajectory_data_idx] = t.to_time()
					trajectory_data_idx += 1

				if topic == '/sinusoid_trajectory_info':
					if msg.data == 'sleeping':
						hf['command_data'][command_data_idx] = -1
					elif msg.data == 'returning':
						hf['command_data'][command_data_idx] = -2
					else:
						hf['command_data'][command_data_idx] = float(msg.data) # period

					hf['command_data_timestamp'][command_data_idx] = t.to_time()
					command_data_idx += 1 	

				if topic == '/IMU_time':
					#import ipdb;ipdb.set_trace()
					hf['IMU_time'][IMU_time_idx] = t.to_time()
					IMU_time_idx += 1

				if topic == '/IMU_acc_data':
					hf['IMU_acc_x_data'][IMU_acc_data_idx] = msg.x
					hf['IMU_acc_y_data'][IMU_acc_data_idx] = msg.y
					hf['IMU_acc_z_data'][IMU_acc_data_idx] = msg.z
					
					IMU_acc_data_idx += 1

				if topic == '/IMU_gyro_data':
					#import ipdb;ipdb.set_trace()
					hf['IMU_gyro_x_data'][IMU_gyro_data_idx] = msg.x
					hf['IMU_gyro_y_data'][IMU_gyro_data_idx] = msg.y
					hf['IMU_gyro_z_data'][IMU_gyro_data_idx] = msg.z
					
					IMU_gyro_data_idx += 1

				

			out.release()
			#import ipdb;ipdb.set_trace()
			
			hf.flush()
			hf.close()
			