import re
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def create_z_stack(path, x1, x2, y1, y2, alpha=1):
    global num_stacks

    Z_stack = np.array([])
    num_slices = 0

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
                img = cv2.imread(img_name)[:, :, 0]
                img = cv2.bilateralFilter(img, 5, 50, 50)
                img = cv2.equalizeHist(img)
                img = cv2.erode(img, kernel=cv2.getStructuringElement(cv2.MORPH_ERODE, (3, 3)))
                if Z_stack.shape[0] == 0:
                    Z_stack = np.expand_dims(img, axis=0)
                else:
                    Z_stack = np.vstack((Z_stack, np.expand_dims(img, axis=0)))
            print("Progress: [%d%%]\r" % (((ind + 1) / (1.0 * num_slices)) * 100))

        np.save(path + '/0Z_STACK.npy', Z_stack, allow_pickle=True)

    num_slices = Z_stack.shape[0]
    lateral_cs1 = np.array(Z_stack[:, :, (x1 + x2) // 2])
    lateral_cs2 = np.array(Z_stack[:, (y1 + y2) // 2, :])

    roi1 = lateral_cs1[:, y1:y2]
    roi2 = lateral_cs2[:, x1:x2]
    roi1 = cv2.bilateralFilter(roi1, 3, 50, 50)
    roi2 = cv2.bilateralFilter(roi2, 3, 50, 50)
    ret, thresh_roi1 = cv2.threshold(roi1, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    ret, thresh_roi2 = cv2.threshold(roi2, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    window_size1 = int(alpha * thresh_roi1.shape[1] - 1)
    window_size2 = int(alpha * thresh_roi2.shape[1] - 1)

    thresh_roi1[:, window_size1:thresh_roi1.shape[1] - 1 - window_size1] = 0
    thresh_roi2[:, window_size2:thresh_roi2.shape[1] - 1 - window_size2] = 0

    analyse_stack1 = thresh_roi1[:, 0:window_size1]
    analyse_stack2 = thresh_roi2[:, 0:window_size2]
    analyse_stack3 = thresh_roi1[:, thresh_roi1.shape[1] - window_size1:thresh_roi1.shape[1]]
    analyse_stack4 = thresh_roi2[:, thresh_roi2.shape[1] - window_size2:thresh_roi2.shape[1]]

    analyse_stack1 = (np.sum(analyse_stack1, axis=1) / 255) >= (0.3 * window_size1)
    a_s1_2 = np.roll(analyse_stack1, -1)
    a_s1_3 = np.roll(analyse_stack1, -2)
    stack1 = np.bitwise_and(analyse_stack1, a_s1_2)
    stack1 = np.bitwise_and(stack1, a_s1_3)
    stack1[-1] = True
    val1 = num_slices - np.argmax(np.flip(stack1[:-3], 0))

    analyse_stack2 = (np.sum(analyse_stack2, axis=1) / 255) >= (0.3 * window_size2)
    a_s2_2 = np.roll(analyse_stack2, -1)
    a_s2_3 = np.roll(analyse_stack2, -2)
    stack2 = np.bitwise_and(analyse_stack2, a_s2_2)
    stack2 = np.bitwise_and(stack2, a_s2_3)
    stack2[-1] = True
    val2 = num_slices - np.argmax(np.flip(stack2[:-3], 0))

    analyse_stack3 = (np.sum(analyse_stack3, axis=1) / 255) >= (0.3 * window_size1)
    a_s3_2 = np.roll(analyse_stack3, -1)
    a_s3_3 = np.roll(analyse_stack3, -2)
    stack3 = np.bitwise_and(analyse_stack3, a_s3_2)
    stack3 = np.bitwise_and(stack3, a_s3_3)
    stack3[-1] = True
    val3 = num_slices - np.argmax(np.flip(stack3[:-3], 0))

    analyse_stack4 = (np.sum(analyse_stack4, axis=1) / 255) >= (0.3 * window_size2)
    a_s4_2 = np.roll(analyse_stack4, -1)
    a_s4_3 = np.roll(analyse_stack4, -2)
    stack4 = np.bitwise_and(analyse_stack4, a_s4_2)
    stack4 = np.bitwise_and(stack4, a_s4_3)
    stack4[-1] = True
    val4 = num_slices - np.argmax(np.flip(stack4[:-3], 0))

    z_level = max(val1, val2, val3, val4)

    return lateral_cs1, lateral_cs2, z_level

# try:
#     # z_stack1 = create_z_stack('/home/upamanyu/Documents/Rito/Q-Pi/Data/ctrl001_Data/c1', 1)
#     # z_stack1 = create_z_stack('/home/upamanyu/Documents/Rito/Q-Pi/Data/day5_ctrl_Data/c1', 1)
#     z_stack1 = create_z_stack('/home/upamanyu/Documents/Rito/Q-Pi/Data/no_cells_lam_perl_Data/c2', 1)
#     # z_stack2 = create_z_stack('/home/upamanyu/Documents/Rito/Q-Pi/Data/ctrl001_Data/c2', 2)
#     num_stacks = z_stack1.shape[0]
# except OSError:
#     None
#
# print(z_stack1.shape)
#
# img = cv2.imread('/home/upamanyu/Documents/Rito/Q-Pi/Data/no_cells_lam_perl_Data/c2/37_home.png')
# # img = cv2.imread('/home/upamanyu/Documents/Rito/Q-Pi/Data/ctrl001_Data/c2/44_ctrl001.png')
# # img = cv2.imread('/home/upamanyu/Documents/Rito/Q-Pi/Data/day5_ctrl_Data/c2/63_day5_ctrl.png')
# img_copy = img.copy()
# cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
# cv2.setMouseCallback('Image', handle_opencv_mouse)
# while True:
#     cv2.imshow('Image', img_copy)
#     k = cv2.waitKey(1)
#     if k == ord('x'):
#         break
#
# print(fx, fy, ix, iy)
#
#
# # creating the lateral slice of z stack
# # lateral_cs1 = np.array(z_stack1[:, :, 417, :])
# # lateral_cs2 = np.array(z_stack2[:, :, 417, :])
# #
# # t1 = cv2.cvtColor(lateral_cs1, cv2.COLOR_RGB2GRAY)
# # t2 = cv2.cvtColor(lateral_cs2, cv2.COLOR_BGR2GRAY)
# #
# # temp = cv2.subtract(t2, t1)
# # f = cv2.subtract(t1, temp)
# #
# # f = cv2.applyColorMap(f, cv2.COLORMAP_OCEAN)
# # f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
# # t2 = cv2.applyColorMap(temp, cv2.COLORMAP_OCEAN)
# #
# # final_img = cv2.add(f, t2)
# # plt.imshow(np.vstack((t2, f, final_img)))
# # plt.axvline(694, c='red')
# # plt.show()
# # plt.imshow(lateral_cs)
# # plt.show()
