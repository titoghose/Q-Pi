import re
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def create_z_stack(path, x1, x2, y1, y2, alpha=1):
	Z_stack = np.asarray([None, None])
	
	num_slices = 0

	# search for existing stack, else create
	try:
		print("Trying to load existing Z Stack.")
		Z_stack[0] = np.load(path + 'c1/0Z_STACK_c1.npy')
		Z_stack[1] = np.load(path + 'c2/0Z_STACK_c2.npy')
	except IOError:
		print("Z Stack doesn't exist. Creating now.")

		slices = [[],[]]
		slices[0] = sorted(os.listdir(path+'c1/'), key=lambda z: (int(re.sub('\D', '', z)), z))
		slices[1] = sorted(os.listdir(path+'c2/'), key=lambda z: (int(re.sub('\D', '', z)), z))
		print(np.asarray(slices).shape)
		
		# loops through slices to create stack using np.vstack
		for idx, _ in enumerate(slices):
			f_name = ""
			if idx == 0:
				print("C1 Stack")
				f_name = path + "c1/0Z_STACK_c1.npy"
				img_path = path + "c1/"
			else:
				print("C2 Stack")
				f_name = path + "c2/0Z_STACK_c2.npy"
				img_path = path + "c2/"

			for ind, i in enumerate(slices[idx]):
				num_slices = len(slices[idx])
				if i.endswith('.jpeg') or i.endswith('.png'):
					img_name = img_path + i
					img = cv2.imread(img_name)[:, :, 0]
					img = cv2.bilateralFilter(img, 5, 50, 50)
					# img = cv2.erode(img, kernel=cv2.getStructuringElement(cv2.MORPH_ERODE, (3, 3)))
					# img = cv2.dilate(img, kernel=cv2.getStructuringElement(cv2.MORPH_DILATE, (3, 3)))
					if Z_stack[idx] is None:
						Z_stack[idx] = np.expand_dims(img, axis=0)
					else:
						Z_stack[idx] = np.vstack((Z_stack[idx], np.expand_dims(img, axis=0)))
			
				print("Progress: ", Z_stack[idx].shape, " [%d%%]" % (((ind + 1) / (1.0 * num_slices)) * 100), end="\r")
			
			np.save(f_name, Z_stack[idx], allow_pickle=True)

	num_slices = Z_stack[0].shape[0]

	yz_cross_sec =   {"c1": np.asarray(Z_stack[0][:, :, (x1 + x2) // 2]),
					"c2": np.asarray(Z_stack[1][:, :, (x1 + x2) // 2])} 
	
	xz_cross_sec =   {"c1": np.asarray(Z_stack[0][:, (y1 + y2) // 2, :]),
					"c2": np.asarray(Z_stack[1][:, (y1 + y2) // 2, :])}

	
	roi1 = yz_cross_sec["c1"][:, y1:y2]
	roi2 = xz_cross_sec["c1"][:, x1:x2]
	roi1 = cv2.bilateralFilter(roi1, 3, 50, 50)
	roi2 = cv2.bilateralFilter(roi2, 3, 50, 50)
	ret, thresh_roi1 = cv2.threshold(roi1, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
	ret, thresh_roi2 = cv2.threshold(roi2, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

	window_size1 = int(alpha * thresh_roi1.shape[1] - 1)
	window_size2 = int(alpha * thresh_roi2.shape[1] - 1)

	thresh_roi1[:, window_size1:thresh_roi1.shape[1] - 1 - window_size1] = 0
	thresh_roi2[:, window_size2:thresh_roi2.shape[1] - 1 - window_size2] = 0

	analyse_stack1 = thresh_roi1[:, 0:window_size1]
	analyse_stack2 = thresh_roi2[:, 0:window_size2]
	analyse_stack3 = thresh_roi1[:, thresh_roi1.shape[1] - window_size1:thresh_roi1.shape[1]]
	analyse_stack4 = thresh_roi2[:, thresh_roi2.shape[1] - window_size2:thresh_roi2.shape[1]]

	analyse_stack1 = (np.sum(analyse_stack1, axis=1) / 255) >= (0.3 * window_size1)
	a_s1_2 = np.roll(analyse_stack1, -1)
	a_s1_3 = np.roll(analyse_stack1, -2)
	stack1 = np.bitwise_and(analyse_stack1, a_s1_2)
	stack1 = np.bitwise_and(stack1, a_s1_3)
	stack1[-1] = True
	val1 = num_slices - np.argmax(np.flip(stack1[:-3], 0))

	analyse_stack2 = (np.sum(analyse_stack2, axis=1) / 255) >= (0.3 * window_size2)
	a_s2_2 = np.roll(analyse_stack2, -1)
	a_s2_3 = np.roll(analyse_stack2, -2)
	stack2 = np.bitwise_and(analyse_stack2, a_s2_2)
	stack2 = np.bitwise_and(stack2, a_s2_3)
	stack2[-1] = True
	val2 = num_slices - np.argmax(np.flip(stack2[:-3], 0))

	analyse_stack3 = (np.sum(analyse_stack3, axis=1) / 255) >= (0.3 * window_size1)
	a_s3_2 = np.roll(analyse_stack3, -1)
	a_s3_3 = np.roll(analyse_stack3, -2)
	stack3 = np.bitwise_and(analyse_stack3, a_s3_2)
	stack3 = np.bitwise_and(stack3, a_s3_3)
	stack3[-1] = True
	val3 = num_slices - np.argmax(np.flip(stack3[:-3], 0))

	analyse_stack4 = (np.sum(analyse_stack4, axis=1) / 255) >= (0.3 * window_size2)
	a_s4_2 = np.roll(analyse_stack4, -1)
	a_s4_3 = np.roll(analyse_stack4, -2)
	stack4 = np.bitwise_and(analyse_stack4, a_s4_2)
	stack4 = np.bitwise_and(stack4, a_s4_3)
	stack4[-1] = True
	val4 = num_slices - np.argmax(np.flip(stack4[:-3], 0))

	z_level = max(val1, val2, val3, val4)

	return yz_cross_sec, xz_cross_sec, z_level