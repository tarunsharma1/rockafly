import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt


#path = '/home/mantis/Desktop/top_view_data/camera_streams/UX-S00-top-view/Nov-2-fly6/frames_visualized/'
path = '/home/tarun/Desktop/top_view_data/camera_streams/UX-S333-top-view/Nov-10-fly3/frames_visualized/'


#pickle_file = path + 'final_tests_fly1_2021-11-02-14-27-29_head_roll.p'
pickle_file = path + 'final_tests_fly1_2021-11-10-13-28-23_head_yaw.p'

#vid_capture = cv2.VideoCapture(path + 'final_tests_fly1_2021-11-02-14-27-29_topDLC_resnet50_top_view_headOct25shuffle1_150000_labeled.mp4')
vid_capture = cv2.VideoCapture(path + 'final_tests_fly1_2021-11-10-13-28-23DLC_resnet50_yaw-trackerFeb14shuffle1_150000_filtered_labeled.mp4')


##### convert a specified video to frames in order to visualize ###########

# frame_number = 0
# while(vid_capture.isOpened()):
# 	# vid_capture.read() methods returns a tuple, first element is a bool 
# 	# and the second is frame
# 	ret, frame = vid_capture.read()
# 	if ret == True:
# 		cv2.imwrite(path + str(frame_number) + '.png', frame)
# 		frame_number += 1

# vid_capture.release()

################################################################################

##### from the csv file get frame numbers where roll is a specified value ######


roll_angles = pickle.load(open(pickle_file, "rb"))

print (np.min(roll_angles), np.max(roll_angles))

roll_angles = np.array(roll_angles)
## get roll angles in the range -1 and 1
a1 = np.where(roll_angles<=np.min(roll_angles)+2)[0]
a2 = np.where(roll_angles <=np.min(roll_angles)+2)[0]
frames_to_visualize = np.intersect1d(a1,a2)

for frame_number in frames_to_visualize:
	img = cv2.imread(path + str(frame_number) +'.png')
	cv2.imwrite(path + 'results/' + str(frame_number) +'_' +str(roll_angles[frame_number])+ '.png', img)
	