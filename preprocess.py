import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D

# self defined function from createZStack.py
from createZStack import create_z_stack

# variable to store coordinates of bounding box for selected cell
ix, iy, fx, fy = 0, 0, 0, 0

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


def handle_matplotlib_mouse(event):
    global membrane_z
    membrane_z = event.ydata


# initializing OpenCV window and drawing event handling
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", handle_opencv_mouse)

# Loading the image corresponding to z = 30 (middle value) to enable bounding box drawing for cell selection + resizing
img = cv2.imread("/Users/upamanyughose/Documents/Rito/cellVolume/membrane_cell/c2/noLUT_z30c2.jpeg")
img = cv2.resize(img, (480, 480))

# Loop handling drawing of bounding boxes
while True:
    cv2.imshow("Image", img)
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

try:
    os.mkdir("./postProcessing")
    os.mkdir("./contourLines")
except OSError:
    None

# looping through the z slices to extract cell countours in each slice
for ind, i in enumerate(os.listdir("./membrane_cell/c2/")):
    if i.startswith(".") or i.endswith(".npy"):
        continue
    img = cv2.imread(os.getcwd() + "/membrane_cell/c2/" + i)
    img = cv2.resize(img, (480, 480))
    img = img[iy:fy, ix:fx, 0]
    if ("z30" in i) or ("Z30" in i):
        cv2.imwrite("postProcessing/afterCrop.jpg", img)

    im_og = img.copy()

    # applying bilateral filter to preserve edges while removing noise; eg. Gaussian Blur is not good at edge
    # preservation
    filtered = cv2.bilateralFilter(img, 5, 75, 75)
    if ("z30" in i) or ("Z30" in i):
        cv2.imwrite("postProcessing/afterBilateralFilter.jpg", filtered)

    # finding threshold value as a linear function of slice number due to varying intensity of cell brightness in
    # each slice
    thresh_val = 190 - (((ind - 1) / 6) * 9)

    # applying binary threshold based on thresh_val to get rid of background
    ret, img = cv2.threshold(filtered, thresh_val, 255, cv2.THRESH_BINARY)
    if ("z30" in i) or ("Z30" in i):
        cv2.imwrite("postProcessing/afterBinaryThreshold.jpg", img)

    # applying open operation i.e erosion followed by dilation to get rid of small noisy outliers in background
    kernel = np.ones((3, 3))
    img = cv2.morphologyEx(img, op=cv2.MORPH_OPEN, kernel=kernel)
    if ("z30" in i) or ("Z30" in i):
        cv2.imwrite("postProcessing/afterOpen.jpg", img)

    # applying close operation i.e dilation followed by erosion to regenerate portions of cell that might have been
    # lost due to opening operarion
    kernel = np.ones((3, 3))
    img = cv2.morphologyEx(img, op=cv2.MORPH_CLOSE, kernel=kernel)
    if ("z30" in i) or ("Z30" in i):
        cv2.imwrite("postProcessing/afterClose.jpg", img)

    im = img.copy()

    # finding contours (array of 2d coordinates) of cell
    _, contours, hierarchy = cv2.findContours(im, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_TC89_L1)

    # converting image to color to enable drawing of contours
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # creating blank image to draw contours of cell on
    img_temp = np.zeros(img.shape, dtype='uint8')

    # initializing array to store new contours after filtering out ones with too less area i.e noise
    new_contours = np.array([])
    maxArea = 0
    maxInd = -1

    # loop to remove too small contours
    for x, cont in enumerate(contours):
        if cv2.contourArea(cont) > maxArea and cv2.contourArea(cont) > 50:
            maxArea = cv2.contourArea(cont)
            maxInd = x

    # removing extra dimensions from countour array
    if len(contours) != 0 and maxInd != -1:
        new_contours = np.squeeze(np.array(contours[maxInd]))
        img_cont = cv2.drawContours(img_temp, [new_contours], -1, (255, 255, 255), 1)
        cv2.imwrite("contourLines/cont_" + i, img_cont)
        if ("z30" in i) or ("Z30" in i):
            cv2.imwrite("postProcessing/initialContours.jpg", img_cont)

    # fitting the closest ellipse (approximation) to the contours in order to take care of cell boundaries that might
    # not have been picked up
    if new_contours.shape[0] != 0:
        ellipse = cv2.fitEllipse(new_contours)
        img_temp = np.zeros(img.shape, dtype='uint8')
        img = cv2.ellipse(img_temp, ellipse, (255, 255, 255), -1)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        im = img.copy()

        _, contours, hierarchy = cv2.findContours(im, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        img_temp = np.zeros(img.shape, dtype='uint8')
        img = cv2.drawContours(img_temp, contours, 0, (255, 255, 255), 1)

        cv2.imwrite("contourLines/cont_ellipse_" + i, img)
        if ("z30" in i) or ("Z30" in i):
            cv2.imwrite("postProcessing/minEllipseContours.jpg", img)

        new_contours = np.squeeze(np.array(contours))

        if len(new_contours) != 0:
            new_contours = np.insert(new_contours, 2, (ind + 1), axis=1)
            if final_contours.shape[0] == 0:
                final_contours = new_contours
            else:
                final_contours = np.vstack((final_contours, new_contours))
    else:
        img = img_temp

# initializing figure for wiremesh reconstruction
fig3d = plt.figure()

ax1 = Axes3D(fig3d)

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

# applying Convex Hull algorithm to set of contour points to find minimum convex boundary of all points
ch = ConvexHull(final_contours)

# Calculating volume of entire cell
x_factor = (2048 / 480.) * 0.06905
y_factor = (2048 / 480.) * 0.06905
z_factor = 0.2
print "Entire Volume: ", ch.volume * x_factor * y_factor * z_factor

# finding stack to display in order to get z = membrane
z_stack = create_z_stack('/Users/upamanyughose/Documents/Rito/cellVolume/membrane_cell/c2')

# finding centroid of reconstructed cell
cx = int(round(np.mean(ch.points[ch.vertices, 0]), 0))
cy = int(round(np.mean(ch.points[ch.vertices, 1]), 0))
print "Centroid_x: ", cx
print "Centroid_y: ", cy

# creating the lateral slice of z stack
lateral_cs = z_stack[:, cx, :, :]

x = []
y = []
z = []

# plotting wiremesh
for v in ch.simplices:
    x.append(final_contours[v][0])
    y.append(final_contours[v][1])
    z.append(final_contours[v][2])
    ax1.plot(final_contours[v, 0], final_contours[v, 1], final_contours[v, 2], color='blue', antialiased=True)
plt.show()

# figure to show z stack and point out z = membrane
fig = plt.figure()
fig.canvas.mpl_connect('button_press_event', handle_matplotlib_mouse)

plt.imshow(lateral_cs)
plt.show()

membrane_z = round(membrane_z, 0)

print "Membrane Z level: ", membrane_z

# removing all cell contour points above the z = membrane

fc = np.array([])
for ind, pts in enumerate(final_contours):
    if pts[2] >= membrane_z:
        if fc.shape[0] == 0:
            fc = np.expand_dims(pts, axis=0)
        else:
            fc = np.vstack((fc, np.expand_dims(pts, axis=0)))
final_contours = fc

# applying Convex Hull to new set of points under membrane
ch = ConvexHull(final_contours)
print "Vol under membrane: ", ch.volume * x_factor * y_factor * z_factor

fig3d = plt.figure()

ax1 = Axes3D(fig3d)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

# plotting new cell wiremesh below z = membrane
for v in ch.simplices:
    x.append(final_contours[v][0])
    y.append(final_contours[v][1])
    z.append(final_contours[v][2])
    ax1.plot(final_contours[v, 0], final_contours[v, 1], final_contours[v, 2], color='blue', antialiased=True)
plt.show()
