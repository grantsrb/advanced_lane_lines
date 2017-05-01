import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import calibration as calib
import lanes
import filters


def show(img):
    plt.imshow(img)
    plt.show()

cal_imgs_dir = 'camera_cal'
nxy = (9,6) # tuple of chess board corners (across, vertical)
offset = 50

calibrators = calib.calibrate(cal_imgs_dir,nxy)


img = mpimg.imread('test_images/straight_lines2.jpg')
img = calib.undistort(img,calibrators)
filt_img = filters.filter(img)
left_line, right_line = lanes.get_lane_lines(filt_img)
src_pts = np.float32([[left_line[0],left_line[1]],
                    [left_line[2],left_line[3]],
                    [right_line[2],right_line[3]],
                    [right_line[0],right_line[1]]])
# src_pts = np.float32([[277,680],[555,475],[727,474],[1049,680]])
M,M_rev = calib.get_transform(src_pts,img.shape)
img = calib.change_persp(img,M)
show(img)
