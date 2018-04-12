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
            plt.imsave(f + str(j) + '_' + fn.split('/')[1] + '.png', fr, cmap='gray')
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

# Loading the image corresponding middle value of upper and lower z inputs to enable bounding box drawing for cell
# selection + resizing
mid_file_name = mDir + '/c2/'
slices = sorted(os.listdir(mDir + '/c2/'), key=lambda z: (int(re.sub('\D', '', z)), z))

# print slices

for ind, i in enumerate(slices):
    if ind < 45:
        continue

    fn = mid_file_name + i

    img = cv2.imread(fn)
    img_dim = img.shape[1]
    img = cv2.resize(img, (480, 480))

    img = img[:, :, 0]

    filter_img = cv2.bilateralFilter(img, 3, 75, 75)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(filter_img)

    ret2, thresholded_img = cv2.threshold(filter_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thresholded_img[0:1, :] = 0
    thresholded_img[:, 0:1] = 0
    thresholded_img[thresholded_img.shape[0] - 1:thresholded_img.shape[0], :] = 0
    thresholded_img[:, thresholded_img.shape[1] - 1:thresholded_img.shape[1]] = 0

    im_floodfill = thresholded_img.copy()
    h, w = thresholded_img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    floodfill_img = thresholded_img | im_floodfill_inv

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened_img = cv2.morphologyEx(floodfill_img, cv2.MORPH_OPEN, kernel=k, iterations=2)

    sure_bg = cv2.dilate(opened_img, kernel=k, iterations=3)
    dist_transform = cv2.distanceTransform(opened_img, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)
    sure_bg = np.uint8(sure_bg)
    unkown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unkown == 255] = 0
    markers = np.int32(markers)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    cv2.imshow("Img", img)
    cv2.waitKey(0)
