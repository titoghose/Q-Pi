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
    # if ind < 40:
    #     continue

    fn = mid_file_name + i

    img = cv2.imread(fn)
    # img = cv2.resize(img, (480, 480))
    img = img[:, :, 0]

    img = cv2.medianBlur(img, 3)
    # img = cv2.bilateralFilter(img, 7, 75, 75)
    img = cv2.equalizeHist(img)
    img = cv2.erode(img, kernel=cv2.getStructuringElement(cv2.MORPH_ERODE, ksize=(2, 2)), iterations=5)

    img = cv2.bilateralFilter(img, 5, 50, 50)
    ret, thresh_img1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img = thresh_img1

    # cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    # cv2.setMouseCallback('Image', handle_opencv_mouse)
    #
    # while True:
    #     cv2.imshow('Image', img)
    #     k = cv2.waitKey(1)
    #     if k == ord('x'):
    #         break
    #
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    # cv2.waitKey(1)
    # cv2.waitKey(1)
    # cv2.waitKey(1)

    # img = img[iy:fy, ix:fx]
    # print fx, ix, fy, iy
    # print img.shape
    # img = cv2.dilate(img, kernel=cv2.getStructuringElement(cv2.MORPH_DILATE, (2, 2)), iterations=3)
    img = cv2.morphologyEx(img, op=cv2.MORPH_OPEN, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
                           iterations=3)
    img = cv2.dilate(img, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (2, 4)))
    img = cv2.dilate(img, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (4, 2)))

    im_floodfill = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    floodfill_img = img | im_floodfill_inv

    img_temp = np.zeros((floodfill_img.shape[0], floodfill_img.shape[1], 3))
    _, contours, hierarchy = cv2.findContours(floodfill_img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_TC89_L1)

    for cont in contours:
        print cv2.contourArea(cont)
        if cv2.contourArea(cont) < 200:
            img_temp = cv2.drawContours(img_temp, [cont], -1, (255, 255, 255), -1)

    img_temp = np.uint8(img_temp[:, :, 0])
    floodfill_img = np.uint8(floodfill_img)

    final_img = cv2.subtract(floodfill_img, img_temp)

    # plt.imshow(floodfill_img, cmap='gray')
    # plt.show()

    # k_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel=k_morph)
    # close_img = cv2.morphologyEx(open_img, cv2.MORPH_CLOSE, kernel=k_morph)
    #
    # clahe = cv2.createCLAHE(clipLimit=0.2, tileGridSize=(2, 2))
    # eq_img = clahe.apply(close_img)
    # # eq_img = cv2.equalizeHist(close_img)
    #
    # # filter_img = cv2.medianBlur(eq_img, 5)
    # filter_img = cv2.bilateralFilter(img, 5, 50, 50)
    #
    # ret, threshold_img = cv2.threshold(filter_img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    #
    # close_img2 = cv2.morphologyEx(threshold_img, cv2.MORPH_CLOSE,
    #                               kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    #
    #
    # im_floodfill = close_img2.copy()
    # h, w = close_img2.shape[:2]
    # mask = np.zeros((h + 2, w + 2), np.uint8)
    # cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    # im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # floodfill_img = close_img2 | im_floodfill_inv
    #
    # open_img2 = cv2.morphologyEx(threshold_img, cv2.MORPH_OPEN,
    #                              kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    #
    # dist_transform = cv2.distanceTransform(open_img2, cv2.DIST_L2, 5)
    # dist_transform = cv2.medianBlur(dist_transform, 3)
    # ret, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, cv2.THRESH_BINARY)
    #
    # sure_fg = np.uint8(sure_fg)
    #
    # final_img = np.hstack((img, close_img2))
    #
    cv2.namedWindow("Img", cv2.WINDOW_NORMAL)
    cv2.imshow("Img", floodfill_img)
    cv2.waitKey(0)
