import re
import cv2
import os
import sys
import shutil
import numpy as np
from mayavi import mlab
from pims import ND2_Reader
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

calib = 0
img_dim = 0
num_stacks = 0
ix, iy, fx, fy = 0, 0, 0, 0
membrane_z = 0
drawing = False


def extract_from_ND2(file_name, c):
    global calib
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

    try:
        os.mkdir(f)
        for j, fr in enumerate(frames[0]):
            fn = file_name.split('.')
            plt.imsave(f + str(j) + '_' + fn[0] + '.png', fr, cmap='gray')
            print("Progress: [%f]" % ((100.0 * j) / num_slices))

    except OSError:
        None


file_name = sys.argv[1]
fn = file_name.split('.')[0]
mDir = fn + '_Data'

try:
    os.mkdir(mDir + '/')
except OSError:
    None
extract_from_ND2(file_name, 1)
extract_from_ND2(file_name, 2)


# function handling drawing of bounding box for selected cell
def handle_opencv_mouse(event, x, y, flags, params):
    global ix, iy, fx, fy, drawing, img, im, img_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        img_copy = img.copy()
        drawing = True
        ix, iy = x, y

    if event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y
        cv2.rectangle(img_copy, (ix, iy), (fx, fy), (0, 255, 0), 1)


# function to handle matplotlib mouse press event
def handle_matplotlib_mouse(event):
    global membrane_z
    membrane_z = event.ydata


# function to plot meshgrid
def plot_data(contours, cnt2, fname, mem_z, draw_flag):
    global ix, iy

    ch = ConvexHull(contours)

    max_x = np.max(contours[ch.vertices, 0], axis=0)
    max_y = np.max(contours[ch.vertices, 1], axis=0)
    max_z = np.max(contours[ch.vertices, 2], axis=0)

    max_xy = max(max_x, max_y)
    scale_factor = max_xy / float(sys.argv[3])

    print max_x, max_y, max_z

    if draw_flag is False:
        return ch

    try:
        draw = sys.argv[4]

        fig = mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
        s1, s2, s3 = None, None, None
        # plotting new cell wiremesh below z = membrane
        for v in ch.simplices:
            s2 = mlab.plot3d(contours[v, 0], contours[v, 1], (contours[v, 2] * scale_factor), color=(0, 0, 1),
                             tube_radius=0.3)
            s1 = mlab.points3d(contours[v, 0], contours[v, 1], (contours[v, 2] * scale_factor), color=(0, 0, 0),
                               scale_factor=0.5)
        print "plot above"
        try:
            if mem_z != -1:
                max_lim = max(int(max_x), int(max_y))
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
                s1 = mlab.points3d(cnt2[v, 0], cnt2[v, 1], (cnt2[v, 2] * scale_factor), color=(0, 0, 0),
                                   scale_factor=0.5)
                s2 = mlab.plot3d(cnt2[v, 0], cnt2[v, 1], (cnt2[v, 2] * scale_factor), color=(0, 0, 0), tube_radius=0.05)
                s3 = mlab.triangular_mesh(cnt2[v, 0], cnt2[v, 1], (cnt2[v, 2] * scale_factor), [(0, 1, 2)],
                                          mode='point', color=(1, 0, 0), opacity=0.6)
        print "plot below"

        scale_len = (2. * 480.) / (img_dim * 0.20716)

        scale_bar_x = np.array([int(max_lim + max_lim / 6. + 1), int(max_lim + max_lim / 6. + 1)])
        scale_bar_y = np.array([-int(max_lim / 6.), -int(max_lim / 6.) + scale_len])
        scale_bar_z = np.array([0, 0])

        print "Scale Len: ", scale_len

        mlab.plot3d(scale_bar_x, scale_bar_y, scale_bar_z, color=(0, 0, 0), tube_radius=0.1)

        mlab.savefig(fname.split('.')[0] + '.eps')
        mlab.show()
    except IndexError:
        None

    return ch


# function to create a z stack from a set of slices
def create_z_stack(path):
    global num_stacks

    Z_stack = np.array([])

    # search for existing stack, else create
    try:
        print("Trying to load existing Z Stack.")
        Z_stack = np.load(path + '/0Z_STACK.npy')
    except IOError:
        print("Z Stack doesn't exist. Creating now.")

        slices = sorted(os.listdir(path), key=lambda z: (int(re.sub('\D', '', z)), z))
        num_slices = len(slices)
        num_stacks = num_slices
        # loops through slices to create stack using np.vstack
        for ind, i in enumerate(slices):
            if i.endswith('.jpeg') or i.endswith('.png'):
                img_name = path + '/' + i
                img = plt.imread(img_name)[:, :, 0]
                img = cv2.resize(img, (480, 480))
                if Z_stack.shape[0] == 0:
                    Z_stack = np.expand_dims(img, axis=0)
                else:
                    Z_stack = np.vstack((Z_stack, np.expand_dims(img, axis=0)))
            print "Progress: [%d%%]\r" % (((ind + 1) / (1.0 * num_slices)) * 100)

        np.save(path + '/0Z_STACK.npy', Z_stack, allow_pickle=True)

    return Z_stack


# Loading the image corresponding middle value of upper and lower z inputs to enable bounding box drawing for cell
# selection + resizing
mid_file_name = mDir + '/c2/'
slices = sorted(os.listdir(mDir + '/c2/'), key=lambda z: (int(re.sub('\D', '', z)), z))
TargetInd = int((int(sys.argv[3]) - int(sys.argv[2])) * 0.5) + int(sys.argv[2])
print "Target Index: ", TargetInd
for ind, i in enumerate(slices):
    fn = mid_file_name + i

    img = cv2.imread(fn)
    img_dim = img.shape[1]
    img = cv2.resize(img, (480, 480))

    img = img[:, :, 0]

    # cv2.imshow("Img", img)
    # cv2.waitKey(0)

    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2, 2))
    cl1 = clahe.apply(img)

    # cv2.imshow("Img", cl1)
    # cv2.waitKey(0)

    # filter_img = cv2.GaussianBlur(cl1, (3, 3), 0)
    filter_img = cv2.bilateralFilter(cl1, 3, 75, 75)

    # cv2.imshow("Img", filter_img)
    # cv2.waitKey(0)

    ret2, thresholded_img = cv2.threshold(filter_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    original_img = thresholded_img.copy()

    cv2.imshow("Img", thresholded_img)
    cv2.waitKey(0)
