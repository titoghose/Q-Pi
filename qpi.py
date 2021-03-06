import re
import os
import gc
import time
import shutil
import argparse
import datetime
from xml.etree import ElementTree as ETree

import cv2

import numpy as np
from scipy.spatial import ConvexHull

import javabridge as jb
import bioformats as bf
from pims import ND2_Reader

from mayavi import mlab

import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import CheckButtons
from matplotlib.colors import ListedColormap
from matplotlib.widgets import RectangleSelector

from zstack_formation import create_z_stack


cmap = pl.cm.Reds
red_cmap = cmap(np.arange(cmap.N))
red_cmap[:, 0] = np.ones(256)*(255/255)
red_cmap[:, 1] = np.ones(256)*(91/255)
red_cmap[:, 2] = np.ones(256)*(0/255)
red_cmap[:cmap.N//8, -1] = 0
red_cmap[cmap.N//8:, -1] = np.linspace(0.1, 1, cmap.N - cmap.N//8)
red_cmap = ListedColormap(red_cmap)

cmap = pl.cm.Greens
green_cmap = cmap(np.arange(cmap.N))
green_cmap[:, 0] = np.ones(256)*(168/255)
green_cmap[:, 1] = np.ones(256)*(255/255)
green_cmap[:, 2] = np.ones(256)*(4/255)
green_cmap[:cmap.N//8, -1] = 0
green_cmap[cmap.N//8:, -1] = np.linspace(0.1, 1, cmap.N - cmap.N//8)
green_cmap = ListedColormap(green_cmap)

parser = argparse.ArgumentParser()
parser.add_argument("file_name", help="input .ND2 file")
parser.add_argument("-lb", "--lowerbound", help="specify z slice where cell starts", type=int)
parser.add_argument("-ub", "--upperbound", help="specify z slice where cell ends", type=int)
parser.add_argument("-p", "--plot", help="option to plot the 3D reconstructed cell", action="store_true")
parser.add_argument("-i", "--save_intermediate", help="option to save intermediate z slice outputs", action="store_true")
args = parser.parse_args()

# Global Variables
mdir = ""
slices = None
img_dim = 0
calib = 0
num_stacks = 0
ix, iy, fx, fy = 0, 0, 0, 0
membrane_z = 0
drawing = False
cell_num = 0
next_cell_flag = False
target_ind = 0
rect_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
			  (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
			  (0, 128, 128), (128, 0, 128), (48, 130, 245), (60, 245, 210), (40, 110, 170)]
cell_coords_x = [[0, 0]]
cell_coords_y = [[0, 0]]
bounding_box_ax = None
bounding_box_img = None
yz_hline, yz_vline, xz_hline, xz_vline = None, None, None, None
xz_c1, xz_c2, yz_c1, yz_c2 = None, None, None, None
ax1, ax2 = None, None


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
			bf.write_image(f + str(j) + '_' + fn1[0].split('/')[1] + '.png', img1, pixel_type='uint8')
			print("Progress: [%f]" % ((100.0 * j) / num_stacks), end="\r")
	except OSError:
		print("Found slices at:", f)

	gc.collect()

	return img_dim, calib, num_stacks

def line_select_callback(eclick, erelease):
	global cell_coords_x, cell_coords_y, cell_num
	cell_coords_x[cell_num][0], cell_coords_y[cell_num][0] = int(eclick.xdata), int(eclick.ydata)
	cell_coords_x[cell_num][1], cell_coords_y[cell_num][1] = int(erelease.xdata), int(erelease.ydata)
	print("(%d, %d) --> (%d, %d)" % (cell_coords_x[cell_num][0], cell_coords_y[cell_num][0], cell_coords_x[cell_num][1], cell_coords_y[cell_num][1]))

def toggle_selector(event):
	global cell_num, bounding_box_ax

	left_corner = (cell_coords_x[cell_num][0], cell_coords_y[cell_num][0])
	width = cell_coords_x[cell_num][1]-cell_coords_x[cell_num][0]
	height = cell_coords_y[cell_num][1]-cell_coords_y[cell_num][0]
	rect = Rectangle(left_corner, width, height, color='r', fill=False, linestyle='--')
	bounding_box_ax.add_patch(rect)

	if event.key in ['N', 'n']:
		cell_coords_x.append([0, 0])
		cell_coords_y.append([0, 0])
		cell_num += 1
		plt.draw()

	if event.key in ['X', 'x']:
		cell_num += 1
		plt.draw()
		plt.close()

# function to handle matplotlib mouse press event
def handle_matplotlib_mouse(event):
	global membrane_z, yz_hline, xz_hline, ax1, ax2

	if event.inaxes in [ax1, ax2]:
		membrane_z = event.ydata
		yz_hline.remove()
		xz_hline.remove()
		yz_hline = ax1.axhline(y=event.ydata, c='xkcd:white', lw=1.5, linestyle='--')
		xz_hline = ax2.axhline(y=event.ydata, c='xkcd:white', lw=1.5, linestyle='--')
		plt.draw()

def showC1C2(label):
	global xz_c1, xz_c2, yz_c1, yz_c2
	print(label)
	if label == 'Channel 1':
		xz_c1.set_visible(not xz_c1.get_visible())
		yz_c1.set_visible(not yz_c1.get_visible())
	elif label == 'Channel 2':
		xz_c2.set_visible(not xz_c2.get_visible())
		yz_c2.set_visible(not yz_c2.get_visible())
	plt.draw()

# function to plot meshgrid
def plot_data(contours, cnt2, mem_z, draw_flag):
	ch = ConvexHull(contours)

	if draw_flag is False:
		return ch

	normalising_factor_x = 1
	normalising_factor_y = 1

	# if 0.5 * (fx - ix) >= 50 and 0.5 * (fy - iy) >= 50:
	#     normalising_factor_x = 0.5
	#     normalising_factor_y = 0.5
	# else:
	#     normalising_factor_x = 1
	#     normalising_factor_y = 1

	contours[ch.vertices, 0] = contours[ch.vertices, 0] * normalising_factor_x
	contours[ch.vertices, 1] = contours[ch.vertices, 1] * normalising_factor_y

	max_x = np.max(contours[ch.vertices, 0], axis=0)
	max_y = np.max(contours[ch.vertices, 1], axis=0)
	max_z = np.max(contours[ch.vertices, 2], axis=0)

	max_xy = max(max_x, max_y)
	scale_factor = max_xy / upper_bound
	scale_len = (2 / calib) * normalising_factor_y
	max_lim = max(int(max_x), int(max_y))
	print("Scale Len: ", scale_len)

	if args.plot:

		mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
		# plotting new cell wiremesh below z = membrane
		for v in ch.simplices:
			s1 = mlab.plot3d(contours[v, 0], contours[v, 1], (contours[v, 2] * scale_factor), color=(0, 0, 1),
							 tube_radius=0.3)
			s2 = mlab.points3d(contours[v, 0], contours[v, 1], (contours[v, 2] * scale_factor), color=(0, 0, 0),
							   scale_factor=0.5)
		print("plot above")

		try:
			if mem_z != -1:
				xs = range(-int(max_lim / 6.), int(max_lim + max_lim / 6. + 1))
				ys = range(-int(max_lim / 6.), int(max_lim + max_lim / 6. + 1))
				X, Y = np.meshgrid(xs, ys)
				Z1 = np.ones((len(ys), len(ys))) * (mem_z * scale_factor)
				mlab.mesh(X, Y, Z1, color=(0.6, 0.6, 0.6), opacity=1.0)
			print("plot mesentery")

		except Exception as E:
			print(str(E))

		if cnt2 is not None:
			ch2 = ConvexHull(cnt2)
			for v in ch2.simplices:
				s3 = mlab.points3d(cnt2[v, 0], cnt2[v, 1], (cnt2[v, 2] * scale_factor), color=(0, 0, 0),
								   scale_factor=0.5)
				s4 = mlab.plot3d(cnt2[v, 0], cnt2[v, 1], (cnt2[v, 2] * scale_factor), color=(0, 0, 0), tube_radius=0.05)
				s5 = mlab.triangular_mesh(cnt2[v, 0], cnt2[v, 1], (cnt2[v, 2] * scale_factor), [(0, 1, 2)],
										  mode='point', color=(1, 0, 0), opacity=0.6)
		print("plot below")

		scale_bar_x = np.array([int(max_lim + max_lim / 6. + 1), int(max_lim + max_lim / 6. + 1)])
		scale_bar_y = np.array([-int(max_lim / 6.), -int(max_lim / 6.) + scale_len])
		scale_bar_z = np.array([0, 0])
		s6 = mlab.plot3d(scale_bar_x, scale_bar_y, scale_bar_z, color=(0, 0, 0), tube_radius=0.1)

		mlab.show()

		mlab.clf()
		mlab.close()
		mlab.close(all=True)

	gc.collect()

	return ch

def select_mid_file(event):
	global bounding_box_ax, bounding_box_img, target_ind, slices, mdir

	if target_ind + event.step < 0:
		target_ind = 0
	elif target_ind + event.step > num_stacks - 1:
		target_ind = num_stacks - 1
	else:
		target_ind += int(event.step)
	bounding_box_ax.cla()
	img = plt.imread(mDir + '/c2/' + slices[target_ind])
	bounding_box_img = bounding_box_ax.imshow(img, cmap='gray')
	plt.draw()
	gc.collect()


jb.start_vm(class_path=bf.JARS)
file_name = args.file_name
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

if args.lowerbound is not None:
	lower_bound = args.lowerbound
else:
	lower_bound = 0

if args.upperbound is not None:
	upper_bound = args.upperbound
else:
	upper_bound = num_stacks

# Loading the image corresponding middle value of upper and lower z inputs to enable bounding box drawing for cell
# selection + resizing
slices = [x for x in sorted(os.listdir(mDir + '/c2/'), key=lambda z: (int(re.sub('\D', '', z)), z)) if "STACK" not in x]

target_ind = int((upper_bound - lower_bound) * 0.5) + lower_bound
mid_file_name = mDir + '/c2/' + slices[target_ind]

print("Microscope Calibration: ", calib)

img = plt.imread(mid_file_name)
img_dim = img.shape[1]

# setting up image:microscope scale variables
x_factor = calib
y_factor = calib
z_factor = 0.2

# Handling drawing of bounding boxes
img_copy = img.copy()
bbox_fig = plt.figure()
bounding_box_ax = plt.gca()
bounding_box_img = bounding_box_ax.imshow(img_copy, cmap='gray')
toggle_selector.RS = RectangleSelector(bounding_box_ax, line_select_callback, drawtype='box', useblit=True, interactive=True)
toggle_selector.RS.to_draw.set_visible(True)
plt.connect('key_press_event', toggle_selector)
plt.connect('scroll_event', select_mid_file)
plt.show()

# Store selected cell ROIs with attached timestamp
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%d%m%Y_%H%M%S')
bbox_fig.savefig(mDir + '/' + str(timestamp) + '_roi.png', dpi=300)
plt.close(bbox_fig)

save_inter = args.save_intermediate

# loop to handle multiple cells

for cn in range(cell_num):
	final_op_string = ''
	print('CELL NUMBER: %d' % (cn + 1))
	iy = cell_coords_y[cn][0]
	fy = cell_coords_y[cn][1]
	ix = cell_coords_x[cn][0]
	fx = cell_coords_x[cn][1]
	roi_centre = [(ix + fx) / 2, (iy + fy) / 2]
	print('Centre of bounding box: (%3f, %3f)' % (roi_centre[0], roi_centre[1]))
	final_op_string += 'Centre of bounding box: (%3f, %3f)' % (roi_centre[0], roi_centre[1])

	# create directories to save intermediate output
	cell_dir = mDir + '/cell_' + str(ix) + '_' + str(iy)
	intermediate_op_dir = cell_dir + '/intermediateOutputs'
	try:
		os.mkdir(cell_dir)
		os.mkdir(intermediate_op_dir)
	except OSError:
		shutil.rmtree(cell_dir)
		os.mkdir(cell_dir)
		os.mkdir(intermediate_op_dir)

	final_contours = np.array([])
	prev_contour = None


	# looping through the z slices to extract cell countours in each slice
	for ind, i in enumerate(sorted(os.listdir(mDir + '/c2/'), key=lambda z: (int(re.sub('\D', '', z)), z))):
		if i.startswith('.') or i.endswith('.npy') or ind < lower_bound or ind > upper_bound:
			continue

		fig_x = plt.figure()

		img_nocrop = cv2.imread(mDir + '/c2/' + i)
		img = img_nocrop[iy:fy, ix:fx, 0]
		cropped_img = img
		axs1 = fig_x.add_subplot(231)
		axs1.imshow(cv2.resize(cropped_img, (0, 0), fx=3.0, fy=3.0), cmap='gray')

		img = cv2.bilateralFilter(img, 5, 75, 75)
		filtered_img = img
		axs2 = fig_x.add_subplot(232)
		axs2.imshow(cv2.resize(filtered_img, (0, 0), fx=3.0, fy=3.0), cmap='gray')

		img = cv2.equalizeHist(img)
		equ_img = img
		# cv2.imwrite(contourLines_dir + '/histeq_' + i, cv2.resize(equ_img, (0, 0), fx=3.0, fy=3.0))

		ret2, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		thresh_img = img
		axs3 = fig_x.add_subplot(233)
		axs3.imshow(cv2.resize(thresh_img, (0, 0), fx=3.0, fy=3.0), cmap='gray')

		k1 = np.ones((3, 3))
		img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel=k1)
		open_img = img
		axs4 = fig_x.add_subplot(234)
		axs4.imshow(cv2.resize(open_img, (0, 0), fx=3.0, fy=3.0), cmap='gray')

		img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel=k1)
		close_img = img
		axs5 = fig_x.add_subplot(235)
		axs5.imshow(cv2.resize(close_img, (0, 0), fx=3.0, fy=3.0), cmap='gray')

		# finding contours (array of 2d coordinates) of cell
		contours, hierarchy = cv2.findContours(img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_TC89_L1)

		# initializing array to store new contours after filtering out ones with too less area i.e noise
		new_contours = np.array([])
		max_area = 0
		max_ind = -1
		min_dist = 1000000

		# loop to find largest contour
		for x, cont in enumerate(contours):
			c = np.squeeze(cont)
			if cont.shape[0] == 1:
				c = np.expand_dims(c, 0)
			centre = [ix + np.mean(c[:, 0]), iy + np.mean(c[:, 1])]
			# print("Centre of contour: (%3f, %3f)" % (centre[0], centre[1])))

			dist_from_roi_centre = (((roi_centre[0] - centre[0]) ** 2) + ((roi_centre[1] - centre[1]) ** 2)) ** .5

			if cv2.contourArea(cont) > max_area and cv2.contourArea(cont) > 500 and dist_from_roi_centre < min_dist:
				max_area = cv2.contourArea(cont)
				max_ind = x
				min_dist = dist_from_roi_centre

		# print(ind, i, max_area)

		# removing extra dimensions from countour array
		if len(contours) != 0 and max_ind != -1:
			img_temp = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')
			new_contours = np.squeeze(np.array(contours[max_ind]))
			img_cont = cv2.drawContours(cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2BGR), [new_contours], -1, (0, 255, 0),
										1)
			axs6 = fig_x.add_subplot(236)
			axs6.imshow(cv2.resize(img_cont, (0, 0), fx=3.0, fy=3.0))

		# fitting the closest ellipse (approximation) to the contours in order to take care of cell boundaries that might
		# not have been picked up
		if new_contours.shape[0] >= 5:
			ellipse = cv2.fitEllipse(new_contours)

			img_temp = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')
			ellipse_img = cv2.ellipse(img_temp, ellipse, (255, 255, 255), -1)

			img = cv2.cvtColor(ellipse_img, cv2.COLOR_BGR2GRAY)

			contours, hierarchy = cv2.findContours(img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
			img_temp = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')
			ellipse_img = cv2.drawContours(img_temp, contours, 0, (0, 0, 255), 1)

			ellipse_area = cv2.contourArea(contours[0])

			# cv2.imwrite(contourLines_dir + '/ellipse_' + i, cv2.resize(ellipse_img, (0, 0), fx=3.0, fy=3.0))

			new_contours = np.squeeze(np.array(contours))

			if len(new_contours) != 0:
				new_contours = np.insert(new_contours, 2, (ind + 1), axis=1)
				prev_contour = new_contours
				if final_contours.shape[0] == 0:
					final_contours = new_contours
				else:
					final_contours = np.vstack((final_contours, new_contours))

		if save_inter:
			fig_x.savefig(intermediate_op_dir + '/slice_' + i, dpi=300)
		plt.close(fig_x)

	# fitting convex hull on points forming final_contours
	conv_hull_full = plot_data(final_contours, None, -1, False)

	# finding centroid (in entire img_dim x img_dim image) of reconstructed cell
	cx = int(ix + round(np.mean(conv_hull_full.points[conv_hull_full.vertices, 0]), 0))
	cy = int(iy + round(np.mean(conv_hull_full.points[conv_hull_full.vertices, 1]), 0))

	# finding stack to display in order to get z = membrane
	yz_cross_sec = None
	xz_cross_sec = None
	try:
		yz_cross_sec, xz_cross_sec, z_level = create_z_stack(mDir+'/', ix, fx, iy, fy)
		num_stacks = yz_cross_sec["c1"].shape[0]
		membrane_z = z_level
	except OSError:
		print("Issue in z-stack creation")

	print('Membrane Z level selected: %d' % (z_level))
	final_op_string += '\nMembrane Z level selected: %d' % (z_level)
	# figure to show z stack and point out z = membrane
	fig = plt.figure()

	ax1 = fig.add_subplot(211)
	ax1.imshow(np.zeros(yz_cross_sec["c1"].shape), cmap='gray')
	yz_c1 = ax1.imshow(yz_cross_sec["c1"], cmap=green_cmap)
	yz_c2 = ax1.imshow(yz_cross_sec["c2"], cmap=red_cmap)
	yz_vline = ax1.axvline(x=((iy+fy)//2), c='xkcd:red')
	yz_hline = ax1.axhline(y=z_level, c='xkcd:white', lw=1.5, linestyle='--')
	ax1.set_aspect(1.5)

	ax2 = fig.add_subplot(212)
	ax2.imshow(np.zeros(xz_cross_sec["c1"].shape), cmap='gray')
	xz_c1 = ax2.imshow(xz_cross_sec["c1"], cmap=green_cmap)
	xz_c2 = ax2.imshow(xz_cross_sec["c2"], cmap=red_cmap)
	xz_vline = ax2.axvline(x=((ix+fx)//2), c='xkcd:red')
	xz_hline = ax2.axhline(y=z_level, c='xkcd:white', lw=1.5, linestyle='--')
	ax2.set_aspect(1.5)

	rax = plt.axes([0.05, 0.4, 0.1, 0.15])
	check = CheckButtons(rax, ('Channel 1', 'Channel 2'), (True, True))
	check.on_clicked(showC1C2)
	fig.canvas.mpl_connect('button_press_event', handle_matplotlib_mouse)
	plt.show()

	membrane_z = int(round(membrane_z, 0))

	# removing all cell contour points above the z = membrane
	fc = np.array([])
	fc2 = np.array([])
	for ind, pts in enumerate(final_contours):
		if pts[2] <= membrane_z:
			if fc.shape[0] == 0:
				fc = np.expand_dims(pts, axis=0)
			else:
				fc = np.vstack((fc, np.expand_dims(pts, axis=0)))
		if pts[2] > membrane_z:
			if fc2.shape[0] == 0:
				fc2 = np.expand_dims(pts, axis=0)
			else:
				fc2 = np.vstack((fc2, np.expand_dims(pts, axis=0)))

	# applying Convex Hull to new set of points under membrane
	try:
		conv_hull_full = plot_data(final_contours, None, -1, True)
		conv_hull_full = plot_data(final_contours, None, membrane_z, True)
		conv_hull_under_mem = plot_data(fc, fc2, membrane_z, False)

		# Calculating volume (quantitative) data
		tot_vol = conv_hull_full.volume * x_factor * y_factor * z_factor
		vol_under_mem = conv_hull_under_mem.volume * x_factor * y_factor * z_factor

	# Exception in case the volume invasion is 0
	except Exception as e:
		print(str(e))
		tot_vol = conv_hull_full.volume * x_factor * y_factor * z_factor
		vol_under_mem = 0


	final_op_string += '\nTotal Volume: %.3f' % (tot_vol)
	final_op_string += '\nVolume Under Membrane: %.3f' % (vol_under_mem)
	final_op_string += '\nPercentage of cell under membrane: %.2f' % ((vol_under_mem / tot_vol) * 100)
	print(final_op_string)
	with open(cell_dir + '/analysis_output.txt', 'a') as f:
		f.write(final_op_string)
		f.write('\n')

	print()
	print()
	gc.collect()