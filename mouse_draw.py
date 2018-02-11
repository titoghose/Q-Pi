import cv2
import numpy as np

ix, iy = 0, 0
drawing = False


def handle_mouse(event, x, y, flags, params):
    global ix, iy, mat, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    if event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(mat, (ix, iy), (x, y), (0, 255, 0), 1)


mat = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', handle_mouse)

while True:
    cv2.imshow('image', mat)
    cv2.waitKey(1)
