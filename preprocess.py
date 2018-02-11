import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

ix, iy, fx, fy = 0, 0, 0, 0
drawing = False
img = None


def handle_mouse(event, x, y, flags, params):
    global ix, iy, fx, fy, drawing, img, im

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    if event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y
        cv2.rectangle(img, (ix, iy), (fx, fy), (0, 255, 0), 1)


cv2.namedWindow("Image")
cv2.setMouseCallback("Image", handle_mouse)

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

    im2, contours, hierarchy = cv2.findContours(im, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_TC89_L1)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    new_contours = []
    for j, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > 50:
            new_contours.append(c)
    img = cv2.drawContours(img, new_contours, -1, (0, 255, 0), 1)

    try:
        cont = np.array(new_contours[0])
        cont = (cont * 2048 * 0.0695) / 480.
        cont = np.insert(cont, 2, (ind+1) * 0.2, axis=2)
        cont = np.squeeze(cont)
        if final_contours.shape[0] == 0:
            final_contours = cont
        else:
            final_contours = np.vstack((final_contours, cont))
    except IndexError:
        c = 0

    #cv2.imshow("Original", im_og)
    #cv2.imshow("Processed", img)
    #c = cv2.waitKey(0)

    #print ind, " : ", thresh_val, "(", chr(c), ")"

#final_contours = np.array(final_contours)

print final_contours.shape

ch = ConvexHull(final_contours)

'''
fig = plt.figure()
for s in ch.simplices:
    plt.plot(final_contours[s, 0], final_contours[s, 1], final_contours[s, 2], 'r.')

plt.show()
'''

print ch.volume