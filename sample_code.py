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

for ind, i in enumerate(slices):
    if ind < 40:
        continue

    fn = mid_file_name + i

    img = cv2.imread(fn)
    img = cv2.resize(img, (480, 480))
    img = img[:, :, 0]

    k_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel=k_morph)
    close_img = cv2.morphologyEx(open_img, cv2.MORPH_OPEN, kernel=k_morph)

    # clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(4, 4))
    # eq_img = clahe.apply(close_img)

    filter_img = cv2.GaussianBlur(close_img, (5, 5), 0)
    # gradient_img = cv2.Laplacian(filter_img, ddepth=cv2.CV_64F, ksize=15)
    # gradient_img = cv2.Canny()
    sobelx = cv2.Sobel(filter_img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(filter_img, cv2.CV_64F, 0, 1, ksize=5)
    gradient_img = cv2.add(sobelx, sobely)

    final_img = gradient_img

    cv2.imshow("Img", final_img)
    cv2.waitKey(0)
