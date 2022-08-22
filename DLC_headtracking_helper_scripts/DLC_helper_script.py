import deeplabcut
import pathlib

#### filtering predictions : for some reason, unlike as with analyze videos, you can't just give a giant list to the filterpredictions method. 
#### If you do that it looks only in the immeadiate parent folder for the prediction csvs. Hence using multiple for loops you have to call the method multiple times
#### each time for one parent folder.


## roll config file
## yaw config file
config_file = '/home/tarun/Desktop/yaw_tracker_2022/yaw_tracker_2022-tarun-2022-06-27/config.yaml'

def convert_posix_list_to_str_list(posix_list):
	str_list = []
	for i in range(0, len(posix_list)):
		str_list.append(str(posix_list[i]))

	return str_list


## get list of all genotypes
genotypes_list = list(pathlib.Path('/home/tarun/Desktop/yaw_head_data/').glob('*'))
genotypes_list = convert_posix_list_to_str_list(genotypes_list)

### hardcode
#print (genotypes_list)
#import sys
#sys.exit(0)
#genotypes_list = ['/home/tarun/Desktop/yaw_head_data/UXJ88yawneg']

for genotype in genotypes_list:
	print ('########' + genotype + ' ##############')
	genotype = genotype + '/'
	individual_flies = list(pathlib.Path(genotype).glob('*'))
	individual_flies = convert_posix_list_to_str_list(individual_flies)

	for fly in individual_flies:
		fly = fly + '/'
		vids_to_filter = list(pathlib.Path(fly).glob('*.avi'))
		vids_to_filter = convert_posix_list_to_str_list(vids_to_filter)

		print (vids_to_filter)
		## use this list to run DLC filter predictions
		deeplabcut.filterpredictions(config_file, vids_to_filter)

