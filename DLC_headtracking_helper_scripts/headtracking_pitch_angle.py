### code to read DLC tracked points and calculate head roll angle based on tether and head position
import csv
from math import atan
import matplotlib.pyplot as plt
import pickle
import glob
import math
import numpy as np


genotype = 'DAKir-S00-top-view'
flies = glob.glob('/home/tarun/Desktop/top_view_data/camera_streams/' + genotype + '/*')


for fly in flies:
	path = fly
	## from the top camera 
	csv_files = glob.glob(path + '/*roll_tracker*_filtered.csv')
	#print (csv_files)

	for trial in csv_files:
		pitch_values = []
		dist_norm = []
		with open(trial) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			line_count = 0
			for row in csv_reader:
				if line_count == 0 or line_count == 1 or line_count == 2:
					line_count += 1
					continue
				tether_start_x = float(row[1])
				tether_start_y = float(row[2])
				tether_end_x = float(row[4])
				tether_end_y = float(row[5])

				bridge_left_x = float(row[7])
				bridge_left_y = float(row[8])
				bridge_right_x = float(row[10])
				bridge_right_y = float(row[11])
				
				head_top_x = float(row[19])
				head_top_y = float(row[20])

				### This will not be an angle but a delta pitch based on pixel positions.
				### For pitch, I take the average of the distances between each bridge point and the head top (X axis starts from top left corner and increase to right)
				### Baseline subtracting these values should take care of variation in head starting position
				### In order to account for changes in position of fly in Z, or for changes in fly head size, I will normalize by the average size of the bridge i.e distance between bridge left and bridge right

				#val = (bridge_left_x + bridge_right_x)/2.0
				dist_norm.append(math.sqrt((bridge_left_y - bridge_right_y)**2 + (bridge_left_x - bridge_right_x)**2))


				## mean of distances between each bridge point and the head top
				
				d1 = math.sqrt((bridge_left_y - head_top_y)**2 + (bridge_left_x - head_top_x)**2)
				d2 = math.sqrt((bridge_right_y - head_top_y)**2 + (bridge_right_x - head_top_x)**2)
				val = (d1 + d2)/2.0

				
				pitch_values.append(round(val, 2))
				#break

		## normalize for head size
		pitch_values = np.array(pitch_values)
		dist_norm = np.array(dist_norm)

		pitch_values = pitch_values/(np.mean(dist_norm))
		pitch_values = list(pitch_values)
		print (min(pitch_values), max(pitch_values))
		#plt.ylim((0,1))
		#plt.plot(range(len(pitch_values)), pitch_values)
		#plt.show()

		pickle.dump(pitch_values, open(trial.split('_topDLC')[0] + '_head_pitch.p','wb'), protocol=2)