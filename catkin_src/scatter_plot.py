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


#list_of_genotypes = ['UX-S00-new', 'UX-S00','UX-S383','UX-S317', 'UX-J88', 'UX-J79', 'UX-J90','UX-J75','UX-S348', 'UX-S333', 'UX-J287']
#genotype_names = ['UX-S00-new','UX-S00-old','UX-SS47222' ,'UX-SS47009', 'UX-60B12','UX-31A09','UX-R74B09','UX-R22E04','UX-SS47214', 'UX-SS47215', 'UX-R44B02']

#list_of_genotypes = ['DAKir-S00', 'DAKir-S383', 'DAKir-S320', 'DAKir-S333', 'DAKir-S317', 'DAKir-J75']
#genotype_names = ['Kir-S00','Kir-SS47222', 'Kir-SS44063', 'Kir-SS47215', 'Kir-SS47009', 'Kir-R22E04']



# list_of_genotypes = ['UX-S00-new', 'UX-J88', 'UX-J79', 'UX-J90']
#genotype_names = ['UX-S00-L', 'UX-S00-R', 'UX-J79-L', 'UX-J79-R', 'UX-J90-L', 'UX-J90-R']
genotype_names = ['UXS00', 'UX28C05' , 'UXR60B12', 'UXR31A09', 'UXR74B09']
list_of_genotypes = ['UXS00yawbothdirections', 'UX28C05yawbothdirections', 'UXJ88yawbothdirections', 'UXJ79yawredobothdirections', 'UXJ90yawredobothdirections']
#list_of_genotypes = ['UXS00yawbothdirections-left','UXS00yawbothdirections-right', 'UXJ79yawredobothdirections-left','UXJ79yawredobothdirections-right', 'UXJ90yawredobothdirections-left','UXJ90yawredobothdirections-right']


for speed in ['1','2']:
	x_list = []
	y_list = []
	x_labels = []
	y_conf_min = []
	y_conf_max = []
	for k,genotype in enumerate(list_of_genotypes):

		y_list_temp = pickle.load(open('/home/tarun/catkin_ws/src/trajectories_autostep_ros/src/data_for_scatter_plot/'+ genotype + '-speed-' + speed +'.p', 'rb'))
		#y_list_temp = pickle.load(open('/home/tarun/catkin_ws/src/trajectories_autostep_ros/src/data_for_ipsi_contra/'+ genotype + '-second-half-right-only-speed-' + speed +'.p', 'rb'))
		
		print (genotype)
		#print ([round(i,2) for i in y_list_temp])
		bootstrapped_data = bootstrap(y_list_temp, 10000)
		conf_min, conf_max = confidence_interval(bootstrapped_data)
		y_conf_min.append(conf_min)
		y_conf_max.append(conf_max)

		x_list.append([k]*len(y_list_temp))
		y_list.append(y_list_temp)
		x_labels.append(genotype_names[k])
		

	#print (y_error_min, y_error_max)
	
	plt.scatter(list(itertools.chain.from_iterable(x_list)), list(itertools.chain.from_iterable(y_list)))
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
	plt.ylim(0,45)
	plt.ylabel('Amplitude of sinusoid fits to WBA yaw')
	plt.xlabel('genotypes')
	plt.title('  WBA yaw speed '+ speed)
	plt.show()
	plt.clf()