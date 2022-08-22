## make scatter plot of flies across genotypes
import pickle
import matplotlib.pyplot as plt
import numpy as np
import itertools
import scipy.stats as st

def bootstrap(data, n):
	## randomly resample n times and take the mean each time
	bootstrapped_data = np.zeros(n)
	for i in range(0,n):
		sample = np.random.choice(data, size=len(data))
		bootstrapped_data[i] = np.mean(np.array(sample))
	return bootstrapped_data


def confidence_interval(data):
	## get the 95% confidence interval by getting the 2.5th and 97.5th percentile of the data
	conf_interval = np.percentile(data,[2.5,97.5])
	print (conf_interval)
	return conf_interval[0], conf_interval[1]


campaniform_fields = {'UX-S00-new':{'dscab':0, 'dped':0, 'vscab':0, 'vped':0, 'lcho':0}, 'UX-J88':{'dscab':0, 'dped':5, 'vscab':3, 'vped':5, 'lcho':7}, 'UX-J79':{'dscab':6, 'dped':16, 'vscab':0, 'vped':11, 'lcho':7}, 'UX-J90':{'dscab':16, 'dped':9, 'vscab':0, 'vped':9, 'lcho':0}}


list_of_genotypes = ['UX-S00-new', 'UX-J88', 'UX-J79', 'UX-J90']
genotype_names = ['UX-S00', 'UX-60B12','UX-31A09','UX-R74B09']



for speed in ['1','2']:
	x_list = []
	y_list = []
	x_labels = []
	y_conf_min = []
	y_conf_max = []
	for k,genotype in enumerate(list_of_genotypes):

		y_list_temp = pickle.load(open('/home/tarun/catkin_ws/src/trajectories_autostep_ros/src/data_for_scatter_plot/'+ genotype + '-speed-' + speed +'.p', 'rb'))
		#print (genotype)
		#print (y_list_temp)
		bootstrapped_data = bootstrap(y_list_temp, 10000)
		conf_min, conf_max = confidence_interval(bootstrapped_data)
		y_conf_min.append(conf_min)
		y_conf_max.append(conf_max)

		## number of cells in a field of interest
		x_list.append([campaniform_fields[genotype]['dscab'] + campaniform_fields[genotype]['lcho']] * len(y_list_temp))
		y_list.append(y_list_temp)
		
		x_labels.append(genotype_names[k])
		#import ipdb;ipdb.set_trace()

	plt.xlim(-5,20)
	for i in range(0,len(y_list)):
		 
		plt.plot(x_list[i], y_list[i], 'o', mfc='none')
		plt.plot(x_list[i][0], np.mean(y_list[i]),'o')

	plt.xlabel('Number of cells silenced (dscab + lcho)')
	plt.ylabel('L-R WBA fit')
	plt.ylim(0, 40)
	plt.show()


	####################### IDEA ###################
	## using the stabilization magnitude as Y values and the number of cells in each field as features, do a linear regression fit to see what fields are significant for wing and for head ##




	#################################################


	## plot the means and conf intervals
	#plt.scatter(range(0,len(list_of_genotypes)), [np.mean(y) for y in y_list],c='r')
	
	## convert the confidence interval values into mean - value so as to use matplotlib errbar function
	y_err_lower = []
	y_err_upper = []
	for i,y in enumerate(y_list):
		mean = np.mean(y)
		y_err_lower.append(mean - y_conf_min[i])
		y_err_upper.append(y_conf_max[i] - mean)
	
	plt.errorbar(range(0,len(list_of_genotypes)), [np.mean(y) for y in y_list], yerr=[y_err_lower, y_err_upper], fmt="o", c='r')

			
	plt.xticks(range(0,len(list_of_genotypes)), x_labels, rotation=25)
	plt.ylim(0,40)
	plt.ylabel('Amplitude of sinusoid fits to average stabilization response')
	plt.xlabel('genotypes')
	plt.title(' UX speed '+ speed)
	plt.show()
	plt.clf()