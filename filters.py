import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

def abs_grad(img, orient='x', sobel_kernel=3, thresh=(20,100), sat_img=[]):
    # ** Returns absolute value of pixel gradients in specified direction on image **

    # img - numpy array to detect pixel gradients off of
    # orient - direction of gradient calculations ('x' or 'y')
    # sobel_kernel - int of kernel size for sobel operation
    # thesh - tuple of lower gradient threshold and upper gradient threshold
    # sat_img - numpy array of saturation values for image. Used to speed calculations

    if len(sat_img) == 0:
        sat_img = get_saturation(img)

    if 'x' in orient.lower():
        x,y=1,0
    else:
        x,y=0,1
    sobel = np.absolute(cv2.Sobel(sat_img,cv2.CV_64F,x,y,ksize=sobel_kernel))
    sobel = (sobel*255/np.max(sobel)).astype(np.uint8)
    binary = np.zeros_like(sobel)
    binary[(sobel > thresh[0]) & (sobel < thresh[1])] = 1
    return binary

def mag_grad(img, sobel_kernel=3,thresh=(100,255), sat_img=[]):
    # ** Returns combined magnitude of pixel gradients in both x and y directions on image **

    # img - numpy array to detect pixel gradients off of
    # sobel_kernel - int of kernel size for sobel operation
    # thesh - tuple of lower gradient threshold and upper gradient threshold
    # sat_img - numpy array of saturation values for image. Used to speed calculations

    if len(sat_img) == 0:
        sat_img = get_saturation(img)

    sobelx = np.square(cv2.Sobel(sat_img,cv2.CV_64F,1,0,ksize=sobel_kernel))
    sobely = np.square(cv2.Sobel(sat_img,cv2.CV_64F,0,1,ksize=sobel_kernel))

    sobel = np.sqrt(sobelx+sobely)
    sobel = (sobel*255/np.max(sobel)).astype(np.uint8)

    binary = np.zeros_like(sobel)
    binary[(sobel > thresh[0]) & (sobel < thresh[1])] = 1
    return binary

def dir_grad(img, sobel_kernel=3, thresh=(0,np.pi/2), sat_img=[]):
    # ** Returns angled pixel gradients on image **

    # img - numpy array to detect pixel gradients off of
    # sobel_kernel - int of kernel size for sobel operation
    # thesh - tuple of lower gradient threshold and upper gradient threshold
    # sat_img - numpy array of saturation values for image. Used to speed calculations

    if len(sat_img) == 0:
        sat_img = get_saturation(img)

    sobelx = np.absolute(cv2.Sobel(sat_img, cv2.CV_64F, 1,0,ksize=sobel_kernel))
    sobely = np.absolute(cv2.Sobel(sat_img,cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    sobeld = np.arctan2(sobely,sobelx)

    binary = np.zeros_like(sobeld)
    binary[(sobeld > thresh[0]) & (sobeld < thresh[1])] = 1
    return binary

def get_saturation(img):
    # ** Returns saturation pixel values from given image **
    # img - numpy array of pixels
    hls = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    return hls[:,:,2]

def filter(img, saturate=True, abs_thresh=(10,100), mag_thresh=(35,100)):
    # Returns binary pixels after combining abs_grad(), mag_grad(), and get_saturation() filters **

    # img - numpy array of pixels
    # saturate - optional boolean to run saturation transform
    # abs_thresh - tuple of thresholds for abs_thresh()
    # mag_thresh - tuple of thresholds for mag_thresh()
    
    if saturate:
        sat_img = get_saturation(img)
    else:
        sat_img = img
    a = abs_grad(img,'x',3,abs_thresh,sat_img=sat_img)
    b = mag_grad(img,sobel_kernel=9, thresh=mag_thresh,sat_img=sat_img)
    d = a+b+sat_img/np.max(sat_img)
    d[d <= 1] = 0
    d[d.nonzero()] = 1
    return d
