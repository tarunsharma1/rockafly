### code to read DLC tracked points and calculate head roll angle based on tether and head position
import csv
from math import atan
import matplotlib.pyplot as plt
import pickle
import glob

genotype = 'UXJ88yawpos'
flies = glob.glob('/home/tarun/Desktop/yaw_head_data/' + genotype + '/*')

#print (min(yaw_angles), max(yaw_angles))
#plt.ylim((-40,40))
		

for fly in flies:
	path = fly
	csv_files = glob.glob(path + '/*filtered.csv')
	#print (csv_files)

	for trial in csv_files:
		yaw_angles = []
		with open(trial) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			line_count = 0
			for row in csv_reader:
				if line_count == 0 or line_count == 1 or line_count == 2:
					line_count += 1
					continue
				#head_mid_x = float(row[7])
				#head_mid_y = float(row[8])
				
				#head_top_x = float(row[16])
				#head_top_y = float(row[17])

				head_left_x = float(row[1])
				head_left_y = float(row[2])
				head_right_x = float(row[4])
				head_right_y = float(row[5])

				## for yaw I get the angle of the line joining tether and head top
				#if (head_mid_x - head_top_x) == 0:
				#	slope = 0
				#else:	
				#	slope = (head_mid_y - head_top_y)/(head_mid_x - head_top_x)
				#	slope = -1/slope
				
				slope = (head_right_y - head_left_y)/(head_right_x - head_left_x)
				ret = atan(slope)
				## Convert the angle from radian to degree
				val = (ret * 180) / 3.14159265
				## Print the result
				#print (round(val, 3))
				yaw_angles.append(round(val, 2))
				#break

		#plt.plot(range(len(yaw_angles)), yaw_angles, linewidth=1.0)
		#plt.show()
		pickle.dump(yaw_angles, open(trial.split('DLC')[0] + '_head_yaw.p','wb'), protocol=2)

#plt.show()
