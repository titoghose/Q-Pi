import gc
import re
import cv2
import os
import shutil
import argparse
import datetime
import numpy as np
import time
from mayavi import mlab
from pims import ND2_Reader
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap
from scipy.spatial import ConvexHull
import javabridge as jb
import bioformats as bf
from xml.etree import ElementTree as ETree

cmap = pl.cm.Reds
red_cmap = cmap(np.arange(cmap.N))
red_cmap[:, 0] = np.ones(256)*(255/255)
red_cmap[:, 1] = np.ones(256)*(91/255)
red_cmap[:, 2] = np.ones(256)*(0/255)
red_cmap[:cmap.N//4, -1] = np.linspace(0, 0.1, cmap.N//4)
red_cmap[cmap.N//4:, -1] = np.linspace(0.4, 1, cmap.N - cmap.N//4)
red_cmap = ListedColormap(red_cmap)

cmap = pl.cm.Greens
green_cmap = cmap(np.arange(cmap.N))
green_cmap[:, 0] = np.ones(256)*(168/255)
green_cmap[:, 1] = np.ones(256)*(255/255)
green_cmap[:, 2] = np.ones(256)*(4/255)
green_cmap[:cmap.N//4, -1] = np.linspace(0, 0.1, cmap.N//4)
green_cmap[cmap.N//4:, -1] = np.linspace(0.4, 1, cmap.N - cmap.N//4)
green_cmap = ListedColormap(green_cmap)


def extract_from_ND2(file_name, c):
	global calib, num_stacks
	print(file_name)
	frames = ND2_Reader(file_name)
	calib = float(frames.metadata['calibration_um'])

	f = ''
	if c == 1:
		f = mDir + '/c1/'
	else:
		f = mDir + '/c2/'

	frames.default_coords['c'] = c - 1

	num_slices = frames[0].shape[0]
	num_stacks = num_slices

	try:
		os.mkdir(f)
		print("Extracting Slices for ", f)
		for j, fr in enumerate(frames[0]):
			fn = file_name.split('.')
			# print(f + str(j) + '_' + fn[0].split('/')[1] + '.png')
			# plt.imsave(f + str(j) + '_' + fn[0].split('/')[1] + '.png', fr, cmap='gray')
			bf.write_image(f + str(j) + '_' + fn[0].split('/')[1] + '.ti', fr)
			print("Progress: [%f]" % ((100.0 * j) / num_slices), end="\r")

	except OSError:
		None

def extract_img(file_name, ch):

	meta = bf.get_omexml_metadata(path=file_name)
	meta = meta.encode('ascii', 'ignore')
	mdroot = ETree.fromstring(meta)

	calib = 0
	img_dim = 0
	num_stacks = 0

	for m in mdroot:
		for l in m:
			if l.tag.endswith("Pixels"):
				calib = float(l.attrib['PhysicalSizeY'])
				img_dim = int(l.attrib['SizeX'])
				num_stacks = int(l.attrib['SizeZ'])

	ir = bf.ImageReader(path=file_name)

	os.system("clear")

	f = ''
	if ch == 1:
		f = mDir + '/c1/'
	else:
		f = mDir + '/c2/'

	try:
		os.mkdir(f)
		print("Extracting into for:", f)
		for j in range(num_stacks):
			fn1 = file_name.split('.')
			img1 = ir.read(c=ch - 1, z=j, rescale=False)
			# plt.imsave(f + str(j) + '_' + fn1[0].split('/')[1] + '.png', img1, cmap='gray')
			bf.write_image(f + str(j) + '_' + fn1[0].split('/')[1] + '.png', img1, pixel_type='uint8')
			print("Progress: [%f]" % ((100.0 * j) / num_stacks), end="\r")
	except OSError:
		print("Found slices at:", f)

	return img_dim, calib, num_stacks

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
					
					# plt.imshow(np.zeros((2048, 2048), dtype=int), cmap='gray')
					# plt.imshow(img, cmap=red_cmap)
					# plt.show()
					img = cv2.bilateralFilter(img, 5, 50, 50)
					# img = cv2.equalizeHist(img)
					img = cv2.erode(img, kernel=cv2.getStructuringElement(cv2.MORPH_ERODE, (3, 3)))
					img = cv2.dilate(img, kernel=cv2.getStructuringElement(cv2.MORPH_DILATE, (3, 3)))
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

def handle_matplotlib_mouse(event):
	plt.cla()
	plt.imshow(np.hstack((np.zeros(yz_cross_sec["c1"].shape),np.zeros(yz_cross_sec["c1"].shape))), cmap='gray')
	plt.imshow(np.hstack((yz_cross_sec["c2"], xz_cross_sec["c2"])), cmap=red_cmap)
	plt.imshow(np.hstack((yz_cross_sec["c1"], xz_cross_sec["c1"])), cmap=green_cmap)
	plt.axvline(x=2048, c='xkcd:red')
	plt.axhline(y=event.ydata, c='xkcd:bright yellow', lw=0.5, linestyle='--')
	ax = plt.gca()
	ax.set_aspect(2)
	plt.draw()


jb.start_vm(class_path=bf.JARS)
file_name = "Data\\2036.nd2"
file_name = file_name.replace("\\", "/")
print(file_name)
fn = file_name.split('.')[0]
mDir = fn + '_Data'

# Create directory to store extracted images
try:
	os.mkdir(mDir + '/')
except OSError:
	None

img_dim, calib, num_stacks = extract_img(file_name, 1)
extract_img(file_name, 2)
jb.kill_vm()

yz_cross_sec, xz_cross_sec, z_level = create_z_stack(mDir+'/', 1310, 1544, 1674, 1913)

fig = plt.figure()
fig.canvas.mpl_connect('button_press_event', handle_matplotlib_mouse)
plt.imshow(np.hstack((np.zeros(yz_cross_sec["c1"].shape),np.zeros(yz_cross_sec["c1"].shape))), cmap='gray')
plt.imshow(np.hstack((yz_cross_sec["c2"], xz_cross_sec["c2"])), cmap=red_cmap)
plt.imshow(np.hstack((yz_cross_sec["c1"], xz_cross_sec["c1"])), cmap=green_cmap)
plt.axvline(x=2048, c='xkcd:red')
plt.axhline(y=z_level, c='xkcd:bright yellow', lw=0.5, linestyle='--')
ax = plt.gca()
ax.set_aspect(2)
plt.show()

