import cv2
import numpy as np
import matplotlib.pyplot as plt

ix, iy, fx, fy = 0, 0, 0, 0


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


img = cv2.imread('/home/upamanyu/Documents/Rito/Q-Pi/Data/ctrl001_Data/c2/46_ctrl001.png')
img = img[:, :, 0]
img = cv2.medianBlur(img, 3)
# img = cv2.bilateralFilter(img, 7, 75, 75)
img = cv2.equalizeHist(img)
img = cv2.erode(img, kernel=cv2.getStructuringElement(cv2.MORPH_ERODE, ksize=(2, 2)), iterations=5)

img = cv2.bilateralFilter(img, 5, 50, 50)
ret, thresh_img1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

img = thresh_img1

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Image', handle_opencv_mouse)

while True:
    cv2.imshow('Image', img)
    k = cv2.waitKey(1)
    if k == ord('x'):
        break

cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)

# img = img[iy:fy, ix:fx]
print fx, ix, fy, iy
print img.shape
# img = cv2.dilate(img, kernel=cv2.getStructuringElement(cv2.MORPH_DILATE, (2, 2)), iterations=3)
img = cv2.morphologyEx(img, op=cv2.MORPH_OPEN, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=3)
img = cv2.dilate(img, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)))
img = cv2.dilate(img, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5)))

im_floodfill = img.copy()
h, w = img.shape[:2]
mask = np.zeros((h + 2, w + 2), np.uint8)
cv2.floodFill(im_floodfill, mask, (0, 0), 255)
im_floodfill_inv = cv2.bitwise_not(im_floodfill)
floodfill_img = img | im_floodfill_inv

plt.imshow(floodfill_img, cmap='gray')
plt.show()

# ret, thresh_img1 = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
# blurred_img = cv2.GaussianBlur(img, (15, 15), 0)
# sub = cv2.subtract(img_t, blurred_img)
# final_img = np.hstack((img, thresh_img1))
# plt.imshow(final_img, cmap='gray')
# plt.show()
