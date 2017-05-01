import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import calibration as calib


def show(img):
    plt.imshow(img)
    plt.show()

cal_imgs_dir = 'camera_cal'
img_type = '.jpg'
nxy = (9,6) # tuple of chess board corners (across, vertical)
offset = 50

calibrators = calib.calibrate(cal_imgs_dir,img_type,nxy)

img = mpimg.imread('camera_cal/calibration9.jpg')
show(img)
M,_ = calib.get_transform(img,nxy,calibrators,offset)
img = calib.change_persp(img,M)
show(img)
