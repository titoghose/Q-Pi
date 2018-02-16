import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D

from createZStack import create_z_stack

ix, iy, fx, fy = 0, 0, 0, 0
membrane_z = 0
drawing = False


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
    membrane_z = event.y


cv2.namedWindow("Image")
cv2.setMouseCallback("Image", handle_opencv_mouse)

img = cv2.imread("/Users/upamanyughose/Documents/Rito/cellVolume/membrane_cell/c2/noLUT_z30c2.jpeg")
img = cv2.resize(img, (480, 480))

while True:
    cv2.imshow("Image", img)
    k = cv2.waitKey(1)
    if k == ord('x'):
        break

final_contours = []
final_contours = np.array(final_contours)

for ind, i in enumerate(os.listdir("./membrane_cell/c2/")):
    if i.startswith(".") or i.endswith(".npy"):
        continue
    img = cv2.imread(os.getcwd() + "/membrane_cell/c2/" + i)
    img = cv2.resize(img, (480, 480))
    img = img[iy:fy, ix:fx, 0]

    im_og = img.copy()

    filtered = cv2.bilateralFilter(img, 5, 75, 75)

    thresh_val = 190 - (((ind - 1) / 6) * 9)

    ret, img = cv2.threshold(filtered, thresh_val, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3))
    img = cv2.morphologyEx(img, op=cv2.MORPH_OPEN, kernel=kernel)

    kernel = np.ones((3, 3))
    img = cv2.morphologyEx(img, op=cv2.MORPH_CLOSE, kernel=kernel)

    im = img.copy()

    _, contours, hierarchy = cv2.findContours(im, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_TC89_L1)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img_temp = np.zeros(img.shape, dtype='uint8')

    new_contours = np.array([])
    maxArea = 0
    maxInd = -1
    for x, cont in enumerate(contours):
        if cv2.contourArea(cont) > maxArea and cv2.contourArea(cont) > 50:
            maxArea = cv2.contourArea(cont)
            maxInd = x
    if len(contours) != 0 and maxInd != -1:
        new_contours = np.squeeze(np.array(contours[maxInd]))

    if new_contours.shape[0] != 0:
        ellipse = cv2.fitEllipse(new_contours)
        img = cv2.ellipse(img_temp, ellipse, (255, 255, 255), -1)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        im = img.copy()

        _, contours, hierarchy = cv2.findContours(im, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.drawContours(np.zeros((100, 100, 3), dtype='uint8'), contours, 0, (255, 255, 255),
                               1)

        new_contours = np.squeeze(np.array(contours))

        if len(new_contours) != 0:
            new_contours = np.insert(new_contours, 2, (ind + 1) * 0.2, axis=1)
            if final_contours.shape[0] == 0:
                final_contours = new_contours * ((2048 / 480.) * 0.06905)
            else:
                final_contours = np.vstack((final_contours, new_contours))

    else:
        img = img_temp

fig3d = plt.figure()

ax1 = Axes3D(fig3d)

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

ch = ConvexHull(final_contours)

z_stack = create_z_stack('/Users/upamanyughose/Documents/Rito/cellVolume/membrane_cell/c2')
cx = int(round(np.mean(ch.points[ch.vertices, 0]), 0))
print cx
lateral_cs = z_stack[:, cx, :, :]

x = []
y = []
z = []

# for v in ch.simplices:
#     x.append(final_contours[v][0])
#     y.append(final_contours[v][1])
#     z.append(final_contours[v][2])
#     ax1.plot(final_contours[v, 0], final_contours[v, 1], final_contours[v, 2], color='blue', antialiased=False)
# plt.show()

fig = plt.figure()
fig.canvas.mpl_connect('button_press_event', handle_matplotlib_mouse)

plt.imshow(lateral_cs)
plt.show()

membrane_z = round(membrane_z, 0)

for pts in final_contours:
    if pts[2] < membrane_z:
        final_contours = np.delete(final_contours, pts)

ch = ConvexHull(final_contours)
print ch.volume