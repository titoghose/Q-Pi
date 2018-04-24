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
from scipy.spatial import ConvexHull
import javabridge as jb
import bioformats as bf
from xml.etree import ElementTree as ETree

from zstack_formation import create_z_stack

parser = argparse.ArgumentParser()
parser.add_argument("file_name", help="input .ND2 file")
parser.add_argument("-lb", "--lowerbound", help="specify z slice where cell starts", type=int)
parser.add_argument("-ub", "--upperbound", help="specify z slice where cell ends", type=int)
parser.add_argument("-p", "--plot", help="option to plot the 3D reconstructed cell", action="store_true")
parser.add_argument("-w", "--window",
                    help="fraction of width to consider while auto picking membrane (eg. 1 is 100 percent or 0.5 is 50 percent )",
                    type=float,
                    default=1)
args = parser.parse_args()

img_dim = 0
calib = 0
num_stacks = 0
ix, iy, fx, fy = 0, 0, 0, 0
membrane_z = 0
drawing = False
cell_num = 0
next_cell_flag = False
rect_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
              (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
              (0, 128, 128), (128, 0, 128), (48, 130, 245), (60, 245, 210), (40, 110, 170)]
cell_coords_x = []
cell_coords_y = []
mem_flag = -1


# function to extract images to png format
def extract_from_ND2(file_name, c):
    global calib, num_stacks
    print file_name
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
        print "Extracting Slices for ", f
        for j, fr in enumerate(frames[0]):
            fn = file_name.split('.')
            # print f + str(j) + '_' + fn[0].split('/')[1] + '.png'
            plt.imsave(f + str(j) + '_' + fn[0].split('/')[1] + '.png', fr, cmap='gray')
            print("Progress: [%f]" % ((100.0 * j) / num_slices))

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
        print "Extracting into for:", f
        for j in range(num_stacks):
            fn1 = file_name.split('.')
            img1 = ir.read(c=ch-1, z=j, rescale=False)
            plt.imsave(f + str(j) + '_' + fn1[0].split('/')[1] + '.png', img1, cmap='gray')
            print("Progress: [%f]" % ((100.0 * j) / num_stacks))
    except OSError:
        print "Found slices at:", f

    return img_dim, calib, num_stacks


# function handling drawing of bounding box for selected cell
def handle_opencv_mouse(event, x, y, flags, params):
    global ix, iy, fx, fy, img, img_copy, next_cell_flag, cell_num

    if next_cell_flag:
        img = img_copy
        next_cell_flag = False

    if event == cv2.EVENT_LBUTTONDOWN:
        img_copy = img.copy()
        ix, iy = x, y

    if event == cv2.EVENT_LBUTTONUP:
        fx, fy = x, y
        cv2.rectangle(img_copy, (ix, iy), (fx, fy), rect_color[cell_num], 2)


# function to handle matplotlib mouse press event
def handle_matplotlib_mouse(event):
    global membrane_z, mem_flag
    membrane_z = event.ydata
    mem_flag += 1
    print membrane_z


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
    print "Scale Len: ", scale_len

    if args.plot:

        mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
        # plotting new cell wiremesh below z = membrane
        for v in ch.simplices:
            s1 = mlab.plot3d(contours[v, 0], contours[v, 1], (contours[v, 2] * scale_factor), color=(0, 0, 1),
                             tube_radius=0.3)
            s2 = mlab.points3d(contours[v, 0], contours[v, 1], (contours[v, 2] * scale_factor), color=(0, 0, 0),
                               scale_factor=0.5)
        print "plot above"

        try:
            if mem_z != -1:
                xs = range(-int(max_lim / 6.), int(max_lim + max_lim / 6. + 1))
                ys = range(-int(max_lim / 6.), int(max_lim + max_lim / 6. + 1))
                X, Y = np.meshgrid(xs, ys)
                Z1 = np.ones((len(ys), len(ys))) * (mem_z * scale_factor)
                mlab.mesh(X, Y, Z1, color=(0.6, 0.6, 0.6), opacity=1.0)

            print "plot mesentery"
        except Exception, E:
            print str(E)

        if cnt2 is not None:
            ch2 = ConvexHull(cnt2)
            for v in ch2.simplices:
                s3 = mlab.points3d(cnt2[v, 0], cnt2[v, 1], (cnt2[v, 2] * scale_factor), color=(0, 0, 0),
                                   scale_factor=0.5)
                s4 = mlab.plot3d(cnt2[v, 0], cnt2[v, 1], (cnt2[v, 2] * scale_factor), color=(0, 0, 0), tube_radius=0.05)
                s5 = mlab.triangular_mesh(cnt2[v, 0], cnt2[v, 1], (cnt2[v, 2] * scale_factor), [(0, 1, 2)],
                                          mode='point', color=(1, 0, 0), opacity=0.6)
        print "plot below"

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


jb.start_vm(class_path=bf.JARS)
file_name = args.file_name
fn = file_name.split('.')[0]
mDir = fn + '_Data'

# Create directory to store extracted images
try:
    os.mkdir(mDir + '/')
except OSError:
    None
# extract_from_ND2(file_name, 1)
# extract_from_ND2(file_name, 2)
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
mid_file_name = mDir + '/c2/'
slices = sorted(os.listdir(mDir + '/c2/'), key=lambda z: (int(re.sub('\D', '', z)), z))
TargetInd = int((upper_bound - lower_bound) * 0.5) + lower_bound
print "Target Index: ", TargetInd
for ind, i in enumerate(slices):
    if ind == TargetInd:
        mid_file_name += i
        break

print mid_file_name
print "Microscope Calibration: ", calib

# initializing OpenCV window and drawing event handling
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Image', handle_opencv_mouse)
img = cv2.imread(mid_file_name)
img_dim = img.shape[1]

# setting up image:microscope scale variables
x_factor = calib
y_factor = calib
z_factor = 0.2

# Loop handling drawing of bounding boxes
img_copy = img.copy()
while True:
    cv2.imshow('Image', img_copy)
    k = cv2.waitKey(1)
    if k == ord('x'):
        cell_coords_x.append((ix, fx))
        cell_coords_y.append((iy, fy))
        cell_num += 1
        break
    if k == ord('n'):
        cell_coords_x.append((ix, fx))
        cell_coords_y.append((iy, fy))
        next_cell_flag = True
        cell_num += 1
        if cell_num == 15:
            print "Reached max limit of cells (15)"
            break

# Store selected cell ROIs with attached timestamp
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%d-%m-%Y_%H:%M:%S')
cv2.imwrite(mDir + '/' + str(timestamp) + 'roi.png', img_copy)

# destroying initial OpenCV display window and closing it
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)

# loop to handle multiple cells
for cn in range(cell_num):
    print "CELL NUMBER: ", cn + 1
    iy = cell_coords_y[cn][0]
    fy = cell_coords_y[cn][1]
    ix = cell_coords_x[cn][0]
    fx = cell_coords_x[cn][1]
    roi_centre = [(ix + fx) / 2, (iy + fy) / 2]
    print("Centre of bounding box: (%3f, %3f)" % (roi_centre[0], roi_centre[1]))

    # create directories to save intermediate output
    cell_dir = mDir + '/cell_' + str(ix) + '_' + str(iy) + '/'
    postProcessing_dir = cell_dir + '/postprocessing'
    contourLines_dir = cell_dir + '/contourLines'
    filtered_dir = cell_dir + '/filtered'
    try:
        os.mkdir(cell_dir)
        os.mkdir(postProcessing_dir)
        os.mkdir(contourLines_dir)
        os.mkdir(filtered_dir)
    except OSError:
        shutil.rmtree(cell_dir)
        os.mkdir(cell_dir)
        os.mkdir(postProcessing_dir)
        os.mkdir(contourLines_dir)
        os.mkdir(filtered_dir)

    final_contours = np.array([])
    prev_contour = None
    # looping through the z slices to extract cell countours in each slice
    for ind, i in enumerate(sorted(os.listdir(mDir + '/c2/'), key=lambda z: (int(re.sub('\D', '', z)), z))):
        if i.startswith('.') or i.endswith('.npy') or ind < lower_bound or ind > upper_bound:
            continue
        img_nocrop = cv2.imread(mDir + '/c2/' + i)
        # img = img_nocrop[682:776, 86:174, 0]
        img = img_nocrop[iy:fy, ix:fx, 0]
        cropped_img = img
        # cv2.imwrite(contourLines_dir + '/cropped_img_' + i, cv2.resize(cropped_img, (0, 0), fx=3.0, fy=3.0))

        img = cv2.bilateralFilter(img, 5, 75, 75)
        filtered_img = img
        # cv2.imwrite(contourLines_dir + '/bilateral_' + i, cv2.resize(filtered_img, (0, 0), fx=3.0, fy=3.0))

        img = cv2.equalizeHist(img)
        equ_img = img
        # cv2.imwrite(contourLines_dir + '/histeq_' + i, cv2.resize(equ_img, (0, 0), fx=3.0, fy=3.0))

        # equ = cv2.GaussianBlur(equ, (3, 3), 0
        # plt.hist(equ.ravel(), 256, [0, 256])
        # plt.show()

        ret2, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_img = img
        # cv2.imwrite(contourLines_dir + '/thresh_' + i, cv2.resize(thresh_img, (0, 0), fx=3.0, fy=3.0))

        k1 = np.ones((3, 3))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel=k1)
        open_img = img
        # cv2.imwrite(contourLines_dir + '/open_' + i, cv2.resize(open_img, (0, 0), fx=3.0, fy=3.0))

        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel=k1)
        close_img = img
        # cv2.imwrite(contourLines_dir + '/close_' + i, cv2.resize(close_img, (0, 0), fx=3.0, fy=3.0))

        # finding contours (array of 2d coordinates) of cell
        _, contours, hierarchy = cv2.findContours(img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_TC89_L1)

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
            # print("Centre of contour: (%3f, %3f)" % (centre[0], centre[1]))

            dist_from_roi_centre = (((roi_centre[0] - centre[0]) ** 2) + ((roi_centre[1] - centre[1]) ** 2)) ** .5

            if cv2.contourArea(cont) > max_area and cv2.contourArea(cont) > 500 and dist_from_roi_centre < min_dist:
                max_area = cv2.contourArea(cont)
                max_ind = x
                min_dist = dist_from_roi_centre

        # print ind, i, max_area

        # removing extra dimensions from countour array
        if len(contours) != 0 and max_ind != -1:
            img_temp = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')
            new_contours = np.squeeze(np.array(contours[max_ind]))
            img_cont = cv2.drawContours(cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2BGR), [new_contours], -1, (0, 255, 0),
                                        1)
            # cv2.imwrite(contourLines_dir + '/contour_' + i, cv2.resize(img_cont, (0, 0), fx=3.0, fy=3.0))

        # fitting the closest ellipse (approximation) to the contours in order to take care of cell boundaries that might
        # not have been picked up
        if new_contours.shape[0] >= 5:
            ellipse = cv2.fitEllipse(new_contours)

            img_temp = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')
            ellipse_img = cv2.ellipse(img_temp, ellipse, (255, 255, 255), -1)

            img = cv2.cvtColor(ellipse_img, cv2.COLOR_BGR2GRAY)

            _, contours, hierarchy = cv2.findContours(img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
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

    # fitting convex hull on points forming final_contours
    conv_hull_full = plot_data(final_contours, None, -1, False)

    # finding centroid (in entire img_dim x img_dim image) of reconstructed cell
    cx = int(ix + round(np.mean(conv_hull_full.points[conv_hull_full.vertices, 0]), 0))
    cy = int(iy + round(np.mean(conv_hull_full.points[conv_hull_full.vertices, 1]), 0))

    # finding stack to display in order to get z = membrane
    lateral_cs1 = None
    lateral_cs2 = None
    try:
        lateral_cs1, lateral_cs2, z_level = create_z_stack(mDir + "/c1/", ix, fx, iy, fy, args.window)
        num_stacks = lateral_cs1.shape[0]
    except OSError:
        None

    print 'Membrane Z level selected: ', z_level

    # figure to show z stack and point out z = membrane
    fig = plt.figure()
    fig.canvas.mpl_connect('button_press_event', handle_matplotlib_mouse)

    rep = 1
    # if num_stacks < 80:
    #     rep = 2

    lateral_cs1 = np.repeat(lateral_cs1, rep, axis=0)
    lateral_cs2 = np.repeat(lateral_cs2, rep, axis=0)

    plt.imshow(np.hstack((lateral_cs1, lateral_cs2)), cmap='gray')
    plt.axvline(cy, c='xkcd:red', lw=2.0)
    plt.axvline(img_dim + cx, c='xkcd:red', lw=2.0)
    plt.axhline(z_level, c='xkcd:bright yellow', lw=1.5)
    plt.gca().invert_yaxis()
    plt.show()

    if mem_flag != -1:
        membrane_z = int(round(membrane_z, 0))
    else:
        membrane_z = z_level
    print 'Membrane Z is now: ', membrane_z

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
    except Exception, e:
        print str(e)
        tot_vol = conv_hull_full.volume * x_factor * y_factor * z_factor
        vol_under_mem = 0

    print 'Total Volume: ', tot_vol
    print 'Volume Under Membrane: ', vol_under_mem
    print 'Percentage of cell under membrane: ', ((vol_under_mem / tot_vol) * 100)
    print
    print
    gc.collect()
