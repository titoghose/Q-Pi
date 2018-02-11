import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

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

img = cv2.imread("/Users/upamanyughose/Documents/Rito/cell_Volume/membrane_cell/noLUT_z30c2.jpeg")
img = cv2.resize(img, (480, 480))

while True:
    cv2.imshow("Image", img)
    k = cv2.waitKey(1)
    if k == ord('x'):
        break

for ind, i in enumerate(os.listdir("./membrane_cell/")):
    if i.endswith("c2.jpeg"):
        img = cv2.imread(os.getcwd() + "/membrane_cell/" + i)
        img = cv2.resize(img, (480, 480))
        img = img[iy:fy, ix:fx, 0]

        filtered = cv2.bilateralFilter(img, 9, 75, 75)

        if ind < 18:
            thresh_val = 160
        else:
            thresh_val = 170

        ret, img = cv2.threshold(filtered, thresh_val, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5))
        img = cv2.morphologyEx(img, op=cv2.MORPH_OPEN, kernel=kernel)

        #kernel = np.ones((3, 3))
        #img = cv2.morphologyEx(img, op=cv2.MORPH_CLOSE, kernel=kernel)

        im = img.copy()

        im2, contours, hierarchy = cv2.findContours(im, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_TC89_L1)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        new_contours = []

        for j, c in enumerate(contours):
            approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
            area = cv2.contourArea(c)
            if area > 250:
                new_contours.append(c)

        img = cv2.drawContours(img, new_contours, -1, (0, 255, 0), 1)

        cv2.imshow("img", img)
        cv2.waitKey(0)