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


# initializing OpenCV window and drawing event handling
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', handle_opencv_mouse)

# Loading the image corresponding middle value of upper and lower z inputs to enable bounding box drawing for cell
# selection + resizing
mid_file_name = mDir + '/c2/'
slices = sorted(os.listdir(mDir + '/c2/'), key=lambda z: (int(re.sub('\D', '', z)), z))
TargetInd = int((int(sys.argv[3]) - int(sys.argv[2])) * 0.5) + int(sys.argv[2])
print "Target Index: ", TargetInd
for ind, i in enumerate(slices):
    if ind == TargetInd:
        mid_file_name += i
        break

print mid_file_name
print "Microscope Calibration: ", calib

img = cv2.imread(mid_file_name)
img_dim = img.shape[1]
img = cv2.resize(img, (480, 480))

# plt.imshow(img)
# plt.show()

# setting up image:microscope scale variables
x_factor = (img_dim / 480.) * calib
y_factor = (img_dim / 480.) * calib
z_factor = 0.2

# Loop handling drawing of bounding boxes
img_copy = img.copy()
while True:
    cv2.imshow('Image', img_copy)
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

roi_centre = [(ix + fx) / 2, (iy + fy) / 2] * int(img_dim / 480.)

print("Rectangle coordinates: (%3f, %3f) (%3f, %3f)" % (ix, iy, fx, fy))
print("Centre of bounding box: (%3f, %3f)" % (roi_centre[0], roi_centre[1]))

# histogram_equalization
# clahe


prev_img = None

# looping through the z slices to extract cell countours in each slice
for ind, i in enumerate(sorted(os.listdir(mDir + '/c2/'), key=lambda z: (int(re.sub('\D', '', z)), z))):
    if i.startswith('.') or i.endswith('.npy') or ind < int(sys.argv[2]) or ind > int(sys.argv[3]):
        continue

    img_nocrop = cv2.imread(mDir + '/c2/' + i)
    img_nocrop = cv2.resize(img_nocrop, (480, 480))

    cropped_img = img_nocrop[iy:fy, ix:fx, 0]

    filtered_img = cv2.bilateralFilter(cropped_img, 3, 50, 50)

    equ = cv2.equalizeHist(filtered_img)

    # equ = cv2.GaussianBlur(equ, (3, 3), 0
    # plt.hist(equ.ravel(), 256, [0, 256])
    # plt.show()

    cv2.imwrite(contourLines_dir + '/cnt_0' + i, equ)

    ret2, thresholded_img = cv2.threshold(equ, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    original_img = thresholded_img.copy()

    k1 = np.ones((3, 3))
    thresholded_img = cv2.morphologyEx(thresholded_img, cv2.MORPH_OPEN, kernel=k1)

    # removing any white pixels on border
    thresholded_img[0:3, :] = 0
    thresholded_img[:, 0:3] = 0
    thresholded_img[:, thresholded_img.shape[1] - 3:thresholded_img.shape[1]] = 0
    thresholded_img[thresholded_img.shape[0] - 3:thresholded_img.shape[0], :] = 0

    cv2.imwrite(contourLines_dir + '/cnt_00' + i, thresholded_img)
    # Copy the thresholded image.
    im_floodfill = thresholded_img.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = thresholded_img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    floodfill_img = thresholded_img | im_floodfill_inv

    cv2.imwrite(contourLines_dir + '/cnt_000' + i, floodfill_img)

    img_for_contour = floodfill_img.copy()

    # finding contours (array of 2d coordinates) of cell
    _, contours, hierarchy = cv2.findContours(img_for_contour, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_TC89_L1)

    # converting image to color to enable drawing of contours
    contour_img = cv2.cvtColor(floodfill_img, cv2.COLOR_GRAY2BGR)

    # creating blank image to draw contours of cell on
    img_temp = np.zeros(contour_img.shape, dtype='uint8')

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

        dist_from_roi_centre = (((roi_centre[0] - centre[0]) ** 2) + ((roi_centre[1] - centre[1]) ** 2)) ** .5

        if cv2.contourArea(cont) > max_area and cv2.contourArea(cont) > 50 and dist_from_roi_centre < min_dist:
            max_area = cv2.contourArea(cont)
            max_ind = x
            min_dist = dist_from_roi_centre

    cont_area = max_area
    ellipse_area = 0

    # removing extra dimensions from countour array
    if len(contours) != 0 and max_ind != -1:
        new_contours = np.squeeze(np.array(contours[max_ind]))
        # print ind, ": ", new_contours.shape
        img_cont = cv2.drawContours(img_temp, [new_contours], -1, (255, 255, 255), 1)
        cv2.imwrite(contourLines_dir + '/cont_000' + i, img_cont)

    # fitting the closest ellipse (approximation) to the contours in order to take care of cell boundaries that might
    # not have been picked up
    if new_contours.shape[0] >= 5:
        ellipse = cv2.fitEllipse(new_contours)

        img_temp = np.zeros(contour_img.shape, dtype='uint8')
        ellipse_img = cv2.ellipse(img_temp, ellipse, (255, 255, 255), -1)

        ellipse_img = cv2.cvtColor(ellipse_img, cv2.COLOR_BGR2GRAY)
        im = ellipse_img.copy()

        _, contours, hierarchy = cv2.findContours(im, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        img_temp = np.zeros(contour_img.shape, dtype='uint8')
        ellipse_img = cv2.drawContours(img_temp, contours, 0, (255, 255, 255), 1)

        ellipse_area = cv2.contourArea(contours[0])

        cv2.imwrite(contourLines_dir + '/cont_ellipse_' + i, ellipse_img)

        new_contours = np.squeeze(np.array(contours))

        if len(new_contours) != 0:
            new_contours = np.insert(new_contours, 2, (ind + 1), axis=1)
            if final_contours.shape[0] == 0:
                final_contours = new_contours
            else:
                final_contours = np.vstack((final_contours, new_contours))
    else:
        final_img = img_temp

    print("%d - Difference: %3f" % (ind, ellipse_area - cont_area))

# fitting convex hull on points forming final_contours
conv_hull_full = plot_data(final_contours, None, cell_dir + '/reconstructed_' + str(ix) + '_' + str(iy) + '.png',
                           38, False)

# finding centroid (in entire img_dim x img_dim image) of reconstructed cell
cx = int((ix + round(np.mean(conv_hull_full.points[conv_hull_full.vertices, 0]), 0)) * (img_dim / 480.))
cy = int((iy + round(np.mean(conv_hull_full.points[conv_hull_full.vertices, 1]), 0)) * (img_dim / 480.))

# finding stack to display in order to get z = membrane
try:
    z_stack = create_z_stack(mDir + "/c1/")
    num_stacks = z_stack.shape[0]
except OSError:
    None

# creating the lateral slice of z stack
print 'Centre of cell in z stack: ', cy
lateral_cs = np.array(z_stack[:, :, int(cx * (480. / img_dim))])
lateral_cs = cv2.resize(lateral_cs, (img_dim, num_stacks))

# figure to show z stack and point out z = membrane
fig = plt.figure()
fig.canvas.mpl_connect('button_press_event', handle_matplotlib_mouse)

plt.imshow(lateral_cs, cmap='gray')
plt.axvline(cy, c='red')
plt.show()

# membrane_z = int(round(membrane_z / 2, 0))
membrane_z = int(round(membrane_z, 0))

# new_cs = []
#
# for i, mem in enumerate(lateral_cs):
#     if membrane_z - 15 < i < membrane_z + 15:
#         new_cs.append(mem)
#
# new_cs = np.array(new_cs)
# print new_cs.shape
#
# plt.imshow(new_cs, cmap='gray')
# plt.show()
print 'Membrane Z level selected: ', membrane_z

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

# print fc.shape
# print fc2.shape

# applying Convex Hull to new set of points under membrane
try:
    conv_hull_under_mem = plot_data(fc, fc2, cell_dir + '/underMem_' + str(ix) + '_' + str(iy) + '.png',
                                    membrane_z, False)
    conv_hull_full = plot_data(final_contours, None, cell_dir + '/reconstructed_' + str(ix) + '_' + str(iy) + '.png',
                               membrane_z, True)

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
