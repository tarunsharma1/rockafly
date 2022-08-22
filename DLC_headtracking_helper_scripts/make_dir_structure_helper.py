### script to automatically make hierarchical directory structure to copy over head tracking results ####

import os
import glob
import pathlib
import shutil

destination = '/home/tarun/Desktop/headtracking_results_yaw_2022_new/'
lines = glob.glob('/home/tarun/Desktop/yaw_head_data/*')


for line in lines:
	genotype = line.split('/')[-1]
	### make genotype folder ####
	os.mkdir(destination + genotype)
	print (destination + genotype)
	flies = glob.glob(line + '/*')

	for fly in flies:
		fly_name = fly.split('/')[-1]
		os.mkdir(destination + genotype + '/' + fly_name)
		#print (destination + genotype + '/' + fly_name)
		pickle_files = glob.glob(fly + '/*_head_*.p')
		for pickle_file in pickle_files:
			file = pickle_file.split('/')[-1]
			shutil.copyfile(pickle_file, destination + genotype + '/' + fly_name + '/' + file)
			print (pickle_file, destination + genotype + '/' + fly_name + '/')
