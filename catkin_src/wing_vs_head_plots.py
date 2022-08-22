import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats


def f(x, A, B): # this is your 'straight line' y=f(x)
	return A*x + B



## per genotype, make plots of wing vs head as scatter plot dots

genotype = 'UXS00yawbothdirections'

wba_both_speeds = []
head_both_speeds = []

speeds = ['1','2']
for speed in speeds:
	wba = pickle.load(open('/home/tarun/catkin_ws/src/trajectories_autostep_ros/src/data_for_scatter_plot/'+ genotype + '-speed-' + speed +'.p', 'rb'))
	head = pickle.load(open('/home/tarun/catkin_ws/src/trajectories_autostep_ros/src/data_for_scatter_plot/'+ genotype + '-speed-' + speed +'-head-yaw.p', 'rb'))
	
	wba_both_speeds.append(wba)
	head_both_speeds.append(head)
	


wba_both_speeds = np.array(wba_both_speeds)
head_both_speeds = np.array(head_both_speeds)

wba = wba_both_speeds.flatten()
head = head_both_speeds.flatten()

a, b = np.polyfit(head, wba, 1)
print ('numpy polyfit: ',a,b)

slope, intercept, r, p, se = stats.linregress(head, wba)
slope = round(slope, 3)
intercept = round(intercept, 3)
r = round(r, 3)
p = round(p, 3)
se = round(se, 3)
print ('linregress: ', slope,intercept, r, p, se)

plt.plot(head, wba, 'o')

plt.xlabel('head')
plt.ylabel('wba')
plt.title('UX-R31A09 corr coef: ' + str(r) + ' slope: '+str(slope) + ' standard err: ' + str(se))
plt.ylim(5,45)
plt.xlim(3,20)
plt.plot(head, a*head + b)
plt.show()