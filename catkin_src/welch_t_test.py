import numpy as np
import scipy.stats as stats
import pickle

controls_genotype = 'UXS00yawbothdirections'
genotype = 'UXJ88yawbothdirections'

wba_both_speeds = []
head_both_speeds = []

speeds = ['1','2']
for speed in speeds:
	print ('Speed: ' + speed)
	wba_controls = pickle.load(open('/home/tarun/catkin_ws/src/trajectories_autostep_ros/src/data_for_scatter_plot/'+ controls_genotype + '-speed-' + speed +'.p', 'rb'))
	head_controls = pickle.load(open('/home/tarun/catkin_ws/src/trajectories_autostep_ros/src/data_for_scatter_plot/'+ controls_genotype + '-speed-' + speed +'-head-yaw.p', 'rb'))

	wba = pickle.load(open('/home/tarun/catkin_ws/src/trajectories_autostep_ros/src/data_for_scatter_plot/'+ genotype + '-speed-' + speed +'.p', 'rb'))
	head = pickle.load(open('/home/tarun/catkin_ws/src/trajectories_autostep_ros/src/data_for_scatter_plot/'+ genotype + '-speed-' + speed +'-head-yaw.p', 'rb'))
	print ('data :')
	print (wba_controls)
	print (wba)

	print ('Wings :')
	print(stats.ttest_ind(wba_controls, wba, equal_var = False))
	print (' Head : ')
	print(stats.ttest_ind(head_controls, head, equal_var = False))
