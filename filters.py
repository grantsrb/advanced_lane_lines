import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

def abs_grad(img, orient='x', sobel_kernel=3, thresh=(20,100)):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    if 'x' in orient.lower():
        x,y=1,0
    else:
        x,y=0,1
    sobel = np.absolute(cv2.Sobel(gray,cv2.CV_64F,x,y,ksize=sobel_kernel))
    sobel = (sobel*255/np.max(sobel)).astype(np.uint8)
    binary = np.zeros_like(sobel)
    binary[(sobel > thresh[0]) & (sobel < thresh[1])] = 1
    return binary

def mag_grad(img, sobel_kernel=3,thresh=(100,255)):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    sobelx = np.square(cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel))
    sobely = np.square(cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel))

    sobel = np.sqrt(sobelx+sobely)
    sobel = (sobel*255/np.max(sobel)).astype(np.uint8)

    binary = np.zeros_like(sobel)
    binary[(sobel > thresh[0]) & (sobel < thresh[1])] = 1
    return binary

def dir_grad(img, sobel_kernel=3, thresh=(0,np.pi/2)):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1,0,ksize=sobel_kernel))
    sobely = np.absolute(cv2.Sobel(gray,cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    sobeld = np.arctan2(sobely,sobelx)

    binary = np.zeros_like(sobeld)
    binary[(sobeld > thresh[0]) & (sobeld < thresh[1])] = 1
    return binary

def filter(img):
    a = abs_grad(img,'x',3,(20,100))
    b = mag_grad(img,sobel_kernel=9, thresh=(30, 100))
    c = dir_grad(img,thresh=(.79,3*np.pi/8-.08))
    d = a+b+c
    d[d <= 1] = 0
    return d
