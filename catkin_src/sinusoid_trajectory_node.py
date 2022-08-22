#!/usr/bin/env python

from __future__ import print_function
import sys
import time
import scipy
import roslib
import rospy
import numpy as np
import std_msgs.msg
from autostep_proxy import AutostepProxy
from autostep_ros.msg import TrackingData
from autostep import Autostep
from std_msgs.msg import String
import matplotlib.pyplot as plt
import os


#autostep = AutostepProxy()


### TODO : Add the ramp so that the sinusoid is damped in the begining and end ###
### also it seems like velocity should be sinusoidal not position ###
### this should publish to a topic called current_position_of_arm in order to bag stuff
### make a launch file for this which first launches this and then launches rosbag record /kinefly_vars current_position_of_arm 



def calibrate():
	print ('calibrating')

	port = '/dev/ttyACM0'

	stepper = Autostep(port)
	stepper.set_gear_ratio(1.0)
	stepper.set_step_mode('STEP_FS_128')
	stepper.set_fullstep_per_rev(200)
	#stepper.set_move_mode_to_max()
	stepper.run_autoset_procedure()
	stepper.calibrate_sensor(360)
	stepper.save_sensor_calibration('sensor3.cal')
	
	return
	import sys;sys.exit(0)

	print('* testing sinusoid')
	sys.stdout.flush()

	# alanas parameters
	# T = 1/f
	#period_list = [3.57,2.78,2.33,2.00,1.75,1.56]#1.41,1.28,1.18,1.09,1.00,0.93] #changed amplitude so recalculate period to get same peak velocity
	period_list = [28.124, 22.5, 18.75, 16.07, 14.06, 12.5, 11.25, 10.23, 9.37]
	## not sure what is offset



	pub = rospy.Publisher('sinusoid_trajectory_info', String, queue_size=10)
	rospy.init_node('sinusoid_trajectory_node', anonymous=True)



	for period in period_list:
		#print (period)
		param = { 
		    'amplitude': 90,
		    'period':  period,
		    'phase':  90,
		    'offset': 0.0, 
		    'num_cycle': 5 
		    }

		stepper.move_to_sinusoid_start(param)
		stepper.busy_wait()
		time.sleep(3.0)

		# need to publish at 200Hz lets say. But instead of time.sleep during which nothing can happen, maybe we can write a loop to publish 'waiting' 
		# for those 3-5 seconds during which the flys delta WBA comes back to normal

		## cannot publish current position when doing sinusoid (see autostep function)
		## Should publish when sinusoid is starting along with the sinusoid params or period
		pub.publish(str(period))
		data = stepper.sinusoid(param)
		stepper.busy_wait()


def trajectory_autostep_proxy():
	# use the proxy node because then we will be getting the position information on the topic motion_data
	# Wil said that the proxy is the interface between autostep_ros and autostep..and that how the motor works is that its a control loop wherein it divides the 
	# trajectory into certain discrete steps, i think thats what autostepproxy.Dt is....and then its almost like a piecewise linear fit to each of these where
	# it checks in each piece if it is behind the position it was supposed to be at (by counting steps) or ahead and automatically adjusts the velocity so that 
	# it matches with set trajectory. 
	#dt = AutostepProxy.TrajectoryDt
	

	## change jog mode params to make it more smooth
	## publish when sinusoid is starting

	autostep = AutostepProxy()
	
	num_cycle = 5
	#period = 3.0
	amplitude = 720
	phase = 90
	rospy.logwarn(' 5 second baseline period ')
	time.sleep(5)
	
	#period_list = [28.124, 22.5, 18.75, 16.07, 14.06, 12.5, 11.25, 10.23, 9.37]
	period_list = [28.124, 22.5, 18.75, 16.07]
	#period_list = [10.23, 9.37]
	pub = rospy.Publisher('sinusoid_trajectory_info', String, queue_size=10)
	rospy.init_node('sinusoid_trajectory_node', anonymous=True)

	for period in period_list:
		rospy.logwarn('period '+ str(period))
		param = { 
			    'amplitude': amplitude,
			    'period':  period,
			    'phase':  phase,
			    'offset': 0.0, 
			    'num_cycle': num_cycle 
			    }
		# Run Trajectory
		autostep.set_move_mode('jog')
		autostep.move_to_sinusoid_start(param)
		autostep.busy_wait()
		#print (stepper.get_position_sensor())

		print ('sleeping')
		pub.publish('sleeping')
		# sleep until fly comes back to normal WBA
		time.sleep(15.0)


		autostep.set_move_mode('max')
		# publish the period at the start of the sinusoid
		pub.publish(str(period))

		autostep.sinusoid(param)
		autostep.busy_wait()

		pub.publish('returning')

		rospy.logwarn('  returning to zero ')
		autostep.set_move_mode('jog')
		autostep.move_to(0.0)
		autostep.busy_wait()


		#print (stepper.get_position_sensor())

		#autostep.run_trajectory(p)
		#while autostep.is_busy():
		#    print('running')
		#    time.sleep(0.1)
		#print('done')







if __name__ == '__main__':
	try:
		trajectory_autostep_proxy()
		#trajectory()
		# terminate the bagging
		os.system("rosnode kill /rosbag_record_trajectory_and_kinefly")
		#os.system("rosnode kill /rosbag_record_camera_stream")
	except rospy.ROSInterruptException:
		pass
