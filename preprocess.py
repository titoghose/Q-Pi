import re
import cv2
import os
import sys
from pims import ND2_Reader
import matplotlib.pyplot as plt
import numpy as np
import shutil
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D


def extract_from_ND2(file_name, c):
    print file_name
    frames = ND2_Reader(file_name)
    f = ''
    if c == 1:
        f = file_name.split('.')[0] + '_c1/'
    else:
        f = file_name.split('.')[0] + '_c2/'

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
extract_from_ND2(file_name, 1)
extract_from_ND2(file_name, 2)

img_dim = 1024

# variable to store coordinates of bounding box for selected cell
ix, iy, fx, fy = 0, 0, 0, 0

# setting up image:microscope scale variables
x_factor = (img_dim / 480.) * 0.06905
y_factor = (img_dim / 480.) * 0.06905
z_factor = 0.2

# variable to store z = membrane
membrane_z = 0

# flag enabling/disabling drawing of bounding box for selected cell
drawing = False


# function handling drawing of bounding box for selected cell
def handle_opencv_mouse(event, x, y, flags, params):
    global ix, iy, fx, fy, drawing, img, im

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    if event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y
        cv2.rectangle(img, (ix, iy), (fx, fy), (0, 255, 0), 1)


# function to handle matplotlib mouse press event
def handle_matplotlib_mouse(event):
    global membrane_z
    membrane_z = event.ydata


# function to plot meshgrid
def plot_data(contours, fname):
    global ix, iy

    ch = ConvexHull(contours)
    fig3d = plt.figure()
    ax = Axes3D(fig3d)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    centroid_x = np.mean(contours[ch.vertices, 0], axis=0)
    centroid_y = np.mean(contours[ch.vertices, 1], axis=0)
    centroid_z = np.mean(contours[ch.vertices, 2], axis=0)

    # print("Centroid: (%3f, %3f, %3f)" % (ix + centroid_x, iy + centroid_y, centroid_z))

    # plt.tight_layout()

    # plotting new cell wiremesh below z = membrane
    for v in ch.simplices:
        x = [a for a in contours[v][0]]
        y = [a for a in contours[v][1]]
        z = [a for a in contours[v][2]]

        # tupleList = zip(x, y, z)
        # poly3d = [[tupleList[vertices[ix][iy]] for iy in range(len(vertices[0]))] for ix in range(len(vertices))]
        # ax.add_collection3d(Poly3DCollection([zip(x, y, z)]))
        ax.plot(contours[v, 0], contours[v, 1], contours[v, 2], color='blue', antialiased=True)
    plt.savefig(fname)
    plt.show()

    return ch


# function to create a z stack from a set of slices
def create_z_stack(path):
    Z_stack = np.array([])

    # search for existing stack, else create
    try:
        print("Trying to load existing Z Stack.")
        Z_stack = np.load(path + '/0Z_STACK.npy')
    except IOError:
        print("Z Stack doesn't exist. Creating now.")

        slices = sorted(os.listdir(path), key=lambda z: (int(re.sub('\D', '', z)), z))
        num_slices = len(slices)
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


# initializing OpenCV window and drawing event handling
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', handle_opencv_mouse)

# folder_name = 'Day1/'
# file_name = folder_name + '52_ctrl002.nd2.png'

# folder_name = 'membrane_cell/c2/'
# file_name = folder_name + 'noLUT_z30c2.jpeg'

# Loading the image corresponding to z = 30 (middle value) to enable bounding box drawing for cell selection + resizing
mid_file_name = file_name.split('.')[0] + '_c2/'
slices = sorted(os.listdir(file_name.split('.')[0] + '_c2/'), key=lambda z: (int(re.sub('\D', '', z)), z))
for ind, i in enumerate(slices):
    if ind == int((len(slices) * (0.5))):
        mid_file_name += i
        break

print mid_file_name

img = cv2.imread(mid_file_name)
img = cv2.resize(img, (480, 480))

# Loop handling drawing of bounding boxes
while True:
    cv2.imshow('Image', img)
    k = cv2.waitKey(1)
    if k == ord('x'):
        break

# destroying initial OpenCV display window and closing it
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)

# creating numpy array to store final contours of selected cell
final_contours = []
final_contours = np.array(final_contours)

# create directories to save intermediate output
postProcessing_dir = file_name.split('.')[0] + '_postProcessing_' + str(ix) + '_' + str(iy)
contourLines_dir = file_name.split('.')[0] + '_contourLines_' + str(ix) + '_' + str(iy)
try:
    os.mkdir(postProcessing_dir)
    os.mkdir(contourLines_dir)
except OSError:
    None

roi_centre = [(ix + fx) / 2, (iy + fy) / 2]

# print("Rectangle coordinates: (%3f, %3f) (%3f, %3f)" % (ix, iy, fx, fy))
# print("Centre of bounding box: (%3f, %3f)" % (roi_centre[0], roi_centre[1]))

# looping through the z slices to extract cell countours in each slice
for ind, i in enumerate(
        sorted(os.listdir(file_name.split('.')[0] + '_c2/'), key=lambda z: (int(re.sub('\D', '', z)), z))):
    if i.startswith('.') or i.endswith('.npy') or ind < int(sys.argv[2]) or ind > int(sys.argv[3]):
        continue

    # print ind, i
    img_nocrop = cv2.imread(file_name.split('.')[0] + '_c2/' + i)
    img_nocrop = cv2.resize(img_nocrop, (480, 480))

    # applying bilateral filter to preserve edges while removing noise; eg. Gaussian Blur is not good at edge
    # preservation
    filtered = cv2.bilateralFilter(img_nocrop, 5, 75, 75)
    if ('z30' in i) or ('Z30' in i):
        cv2.imwrite(postProcessing_dir + '/afterBilateralFilter.jpg', filtered)

    filtered = filtered[iy:fy, ix:fx, 0]
    # print i
    # plt.imshow(filtered, cmap='gray')
    # plt.show()

    # finding threshold value as a linear function of slice number due to varying intensity of cell brightness in
    # each slice
    ret2, img = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # applying binary threshold based on thresh_val to get rid of background
    # ret, img = cv2.threshold(filtered, ret2, 255, cv2.THRESH_BINARY)

    if ('z30' in i) or ('Z30' in i):
        cv2.imwrite(postProcessing_dir + '/afterBinaryThreshold.jpg', img)

    # l_ix, l_iy, l_fx, l_fy = extend_roi(img)

    # img = img[l_iy:l_fy, l_ix:l_fx, 0]
    # cv2.imshow("win", img)
    # cv2.waitKey(0)
    # if ('z30' in i) or ('Z30' in i):
    #     cv2.imwrite(postProcessing_dir + '/afterCrop.jpg', img)

    im_og = img.copy()

    # applying open operation i.e erosion followed by dilation to get rid of small noisy outliers in background
    kernel = np.ones((3, 3))
    img = cv2.morphologyEx(img, op=cv2.MORPH_OPEN, kernel=kernel)
    if ('z30' in i) or ('Z30' in i):
        cv2.imwrite(postProcessing_dir + '/afterOpen.jpg', img)

    # applying close operation i.e dilation followed by erosion to regenerate portions of cell that might have been
    # lost due to opening operarion
    kernel = np.ones((3, 3))
    img = cv2.morphologyEx(img, op=cv2.MORPH_CLOSE, kernel=kernel)
    if ('z30' in i) or ('Z30' in i):
        cv2.imwrite(postProcessing_dir + '/afterClose.jpg', img)

    im = img.copy()

    # finding contours (array of 2d coordinates) of cell
    _, contours, hierarchy = cv2.findContours(im, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_TC89_L1)

    # converting image to color to enable drawing of contours
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # creating blank image to draw contours of cell on
    img_temp = np.zeros(img.shape, dtype='uint8')

    # initializing array to store new contours after filtering out ones with too less area i.e noise
    new_contours = np.array([])
    max_area = 0
    max_ind = -1
    min_dist = 1000000
    # loop to remove too small contours
    for x, cont in enumerate(contours):
        c = np.squeeze(cont)
        if cont.shape[0] == 1:
            c = np.expand_dims(c, 0)
        centre = [ix + np.mean(c[:, 0]), iy + np.mean(c[:, 1])]
        # print("Centre of contour: (%3f, %3f)" % (centre[0], centre[1]))

        dist_from_roi_centre = (((roi_centre[0] - centre[0]) ** 2) + ((roi_centre[1] - centre[1]) ** 2)) ** .5

        if cv2.contourArea(cont) > max_area and cv2.contourArea(cont) > 50 and dist_from_roi_centre < min_dist:
            max_area = cv2.contourArea(cont)
            max_ind = x
            min_dist = dist_from_roi_centre

    # removing extra dimensions from countour array
    if len(contours) != 0 and max_ind != -1:
        new_contours = np.squeeze(np.array(contours[max_ind]))
        print ind, ": ", new_contours.shape
        img_cont = cv2.drawContours(img_temp, [new_contours], -1, (255, 255, 255), 1)
        cv2.imwrite(contourLines_dir + '/cont_' + i, img_cont)
        if ('z30' in i) or ('Z30' in i):
            cv2.imwrite(postProcessing_dir + '/initialContours.jpg', img_cont)

    # fitting the closest ellipse (approximation) to the contours in order to take care of cell boundaries that might
    # not have been picked up
    if new_contours.shape[0] >= 5:
        ellipse = cv2.fitEllipse(new_contours)
        img_temp = np.zeros(img.shape, dtype='uint8')
        img = cv2.ellipse(img_temp, ellipse, (255, 255, 255), -1)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        im = img.copy()

        _, contours, hierarchy = cv2.findContours(im, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        img_temp = np.zeros(img.shape, dtype='uint8')
        img = cv2.drawContours(img_temp, contours, 0, (255, 255, 255), 1)

        cv2.imwrite(contourLines_dir + '/cont_ellipse_' + i, img)
        if ('z30' in i) or ('Z30' in i):
            cv2.imwrite(postProcessing_dir + '/minEllipseContours.jpg', img)

        new_contours = np.squeeze(np.array(contours))

        if len(new_contours) != 0:
            new_contours = np.insert(new_contours, 2, (ind + 1), axis=1)
            if final_contours.shape[0] == 0:
                final_contours = new_contours
            else:
                final_contours = np.vstack((final_contours, new_contours))
    else:
        img = img_temp

# finding stack to display in order to get z = membrane
try:
    z_stack = create_z_stack(file_name.split('.')[0] + "_c1/")
except OSError:
    None

# fitting convex hull on points forming final_contours
conv_hull_full = plot_data(final_contours,
                           file_name.split('.')[0] + '_reconstructed_' + str(ix) + '_' + str(iy) + '.png')

# finding centroid (in entire img_dim x img_dim image) of reconstructed cell
cx = int((ix + round(np.mean(conv_hull_full.points[conv_hull_full.vertices, 0]), 0)) * (img_dim / 480.))
cy = int((iy + round(np.mean(conv_hull_full.points[conv_hull_full.vertices, 1]), 0)) * (img_dim / 480.))

# creating the lateral slice of z stack
print 'Centre of cell in z stack: ', (img_dim - cy)
lateral_cs = np.array(z_stack[:, int(cx * (480. / img_dim)), :])
# lateral_cs = np.repeat(lateral_cs, 2, 0)

# figure to show z stack and point out z = membrane
#     plt.xlabel('y')
# plt.ylabel('z')
fig = plt.figure()
fig.canvas.mpl_connect('button_press_event', handle_matplotlib_mouse)

plt.imshow(lateral_cs, cmap='gray')
plt.axvline((img_dim - cy) * (480. / img_dim), c='red')
plt.show()

# membrane_z = int(round(membrane_z / 2, 0))
membrane_z = int(round(membrane_z, 0))
print 'Membrane Z level selected: ', membrane_z

# removing all cell contour points above the z = membrane
fc = np.array([])
for ind, pts in enumerate(final_contours):
    if pts[2] <= membrane_z:
        if fc.shape[0] == 0:
            fc = np.expand_dims(pts, axis=0)
        else:
            fc = np.vstack((fc, np.expand_dims(pts, axis=0)))

# applying Convex Hull to new set of points under membrane
conv_hull_under_mem = plot_data(fc, file_name.split('.')[0] + '_underMem_' + str(ix) + '_' + str(iy) + '.png')

# Calculating volume (quantitative) data
tot_vol = conv_hull_full.volume * x_factor * y_factor * z_factor
vol_under_mem = conv_hull_under_mem.volume * x_factor * y_factor * z_factor

print 'Total Volume: ', tot_vol
print 'Volume Under Membrane: ', vol_under_mem
print 'Percentage of cell under membrane: ', ((vol_under_mem / tot_vol) * 100)
