import cv2

vid_name = '/home/tarun/Downloads/final_tests_fly1_2022-06-15-18-25-46.avi'
cap = cv2.VideoCapture(vid_name)
out = cv2.VideoWriter('/home/tarun/Downloads/test.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (500,500))

while(cap.isOpened()):
	ret, frame = cap.read()
	if ret == True:
		# Display the resulting frame
		(h, w) = frame.shape[:2]
		(cX, cY) = (w // 2, h // 2)
		# rotate our image by 45 degrees around the center of the image
		M = cv2.getRotationMatrix2D((cX, cY), 180, 1.0)
		frame = cv2.warpAffine(frame, M, (w, h))
		out.write(frame)
		#cv2.imshow('Frame',frame)
		#cv2.waitKey(30)

	# Break the loop
	else:
		break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
