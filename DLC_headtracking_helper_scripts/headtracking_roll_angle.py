### code to read DLC tracked points and calculate head roll angle based on tether and head position
import csv
from math import atan
import matplotlib.pyplot as plt
import pickle
import glob

genotype = 'DAKir-S00-top-view'
flies = glob.glob('/home/tarun/Desktop/top_view_data/camera_streams/' + genotype + '/*')



for fly in flies:
	path = fly
	csv_files = glob.glob(path + '/*roll_tracker*_filtered.csv')
	#print (csv_files)

	for trial in csv_files:
		roll_angles = []
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
				
				## need to get a line joining the two tether points
				# if (tether_end_x - tether_start_x) == 0:
				# 	## vertical line, undefined slope
				# 	print ('verical line')
				# 	tether_slope = None
					
				# else:	
				# 	tether_slope = (tether_end_y - tether_start_y)/(tether_end_x - tether_start_x)
				# 	tether_intercept = tether_end_y - tether_slope*tether_end_x
				
				### instead of using the tether end points lets try to use perfectly vertical end points on the frame itself.


				## need to get a line joining the two bridge points
				if (bridge_right_x - bridge_left_x) == 0:
					bridge_perpendicular_slope = 0
				else:
					bridge_slope = (bridge_right_y - bridge_left_y)/(bridge_right_x - bridge_left_x)
					## line percendicular to the bridge line
					bridge_perpendicular_slope = -1.0/bridge_slope

				## angle between tether and bridge_perpendicular will give us roll angle
				#angle = (bridge_perpendicular_slope - tether_slope) / (1 + tether_slope * bridge_perpendicular_slope)
				## Calculate tan inverse of the angle
				#ret = atan(angle)
				ret = atan(bridge_perpendicular_slope)
				## Convert the angle from radian to degree
				val = (ret * 180) / 3.14159265
				## Print the result
				#print (round(val, 3))
				roll_angles.append(round(val, 2))
				#break

		print (min(roll_angles), max(roll_angles))
		#plt.ylim((-40,40))
		#plt.plot(range(len(roll_angles)), roll_angles)
		#plt.show()

		pickle.dump(roll_angles, open(trial.split('_topDLC')[0] + '_head_roll.p','wb'), protocol=2)