import glob
import shutil

genotypes = glob.glob('/home/tarun/catkin_ws/src/trajectories_autostep_ros/bagfiles/headtracking_results_yaw_2022_new/*')
#genotypes = ['/home/tarun/catkin_ws/src/trajectories_autostep_ros/bagfiles/headtracking_results_new/DAKir-S00-top-view']
for genotype in genotypes:
	genotype_name = genotype.split('/')[-1]
	flies = glob.glob(genotype + '/*')
	for fly in flies:
		fly_name = fly.split('/')[-1]
		pickle_files = glob.glob(fly + '/*_head_*.p')
		for p in pickle_files:
			file_name = p.split('/')[-1]
			shutil.copyfile(p, '/home/tarun/catkin_ws/src/trajectories_autostep_ros/bagfiles/hdf5/' + genotype_name + '/' + fly_name + '/' + file_name)
			print (p, '/home/tarun/catkin_ws/src/trajectories_autostep_ros/bagfiles/hdf5/' + genotype_name + '/' + fly_name + '/' + file_name)
