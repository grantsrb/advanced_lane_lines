import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import inout

def get_imgpts(img,nxy,img_pts):
    # ** Used to find and return corners on chessboard in image **

    # img - image that contains a chessboard as a numpy array
    # nxy - tuple of integers containing number of corners on chessboard in the x direction and in the y direction.
    # img_pts - array to store corner locations in

    gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    ret,corners = cv2.findChessboardCorners(gray_img,nxy,None)
    if ret:
        img_pts.append(corners)
    return ret,img_pts

def make_objpts(nxy):
    # ** Constructs object points to map to **

    # nxy - tuple of integers containing number of corners on chessboard in the x direction and in the y direction.

    objp = np.zeros((nxy[0]*nxy[1],3),dtype=np.float32)
    objp[:,:2] = np.mgrid[0:nxy[0],0:nxy[1]].T.reshape(-1,2)
    return objp

def draw_corners(img,nxy=(9,6),gray_img=[]):
    # ** Draws lines on the corners of a chessboard **

    # img - image containing a chessboard
    # nxy - tuple of integers containing number of corners on chessboard in the x direction and in the y direction.
    # gray_img - optional gray copy of img to speed calculations

    if len(gray_img) == 0:
        gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray_img,nxy,None)
    if ret:
        cv2.drawChessboardCorners(img,nxy,corners,ret)
    return ret

def calibrate(imgs_dir,nxy):
    # ** Reads in images to calculate the calibration and undistortion parameters **

    # imgs_dir - root directory as string containing calibration images
    # nxy - tuple of integers containing number of corners on chessboard in the x direction and in the y direction.

    cal_paths = inout.read_paths(imgs_dir)
    img_pts = []
    obj_pts = []
    objp = make_objpts(nxy) # locations for transform

    images = []
    for path in cal_paths[1:]:
        img = mpimg.imread(path)
        images.append(img)
        ret,img_pts = get_imgpts(img,nxy,img_pts)
        if ret:
            obj_pts.append(objp)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts,
                                                        img_pts,
                                                        img.shape[:2],
                                                        None,
                                                        None)
    return ret, mtx, dist, rvecs, tvecs

def undistort(img,calibrators):
    # ** Uses calibrator parameters from calibrate() to undistort an image **

    # img - image as numpy array to be undistorted
    # calibrators - tuple of parameters received from cv2.calibrateCamera(), the tuple has the form (ret,mtx,dist,rvecs,tvecs)

    return cv2.undistort(img, calibrators[1],
                        calibrators[2],
                        None,
                        calibrators[1])

def get_chess_transform(img, nxy, calibrators, offset):
    dst = undistort(img,calibrators)
    dst_gray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(dst_gray,nxy,None)
    if ret:
        x,y = nxy

        src_pts = np.float32([corners[0],
                            corners[x-1],
                            corners[-1],
                            corners[-x]])

        img_size = (dst_gray.shape[1],dst_gray.shape[0])
        dst_pts = np.float32([[offset, offset],
                            [img_size[0]-offset, offset],
                            [img_size[0]-offset, img_size[1]-offset],
                            [offset, img_size[1]-offset]])

        M = cv2.getPerspectiveTransform(src_pts,dst_pts)
        M_rev = cv2.getPerspectiveTransform(dst_pts,src_pts)
        return M, M_rev
    return None, None

def get_transform(src_pts, img_size, offset=(250,0),top_offset=20):
    # ** Constructs and returns a perspective transform matrix **

    # src_pts - numpy float32 array of four row,col points on image to be transformed
    # img_size - the shape of the image to be transformed
    # offset - determines perspective transform locations
    # top_offset - parameter used for additional offset from top of new perspective image

    xoff,yoff = offset
    dst_pts = np.float32([[xoff, img_size[0]-yoff],
                        [xoff, yoff+top_offset],
                        [img_size[1]-xoff, yoff+top_offset],
                        [img_size[1]-xoff, img_size[0]-yoff]])
    M = cv2.getPerspectiveTransform(src_pts,dst_pts)
    M_rev = cv2.getPerspectiveTransform(dst_pts,src_pts)
    return M, M_rev

def change_persp(img,M):
    # ** Uses a perspective transform matrix to transform the perspective of the given image **

    # img - image as numpy array to be transformed
    # M - perspective transform matrix as numpy array
    
    warped = cv2.warpPerspective(img,M,
                        dsize=(img.shape[1],img.shape[0]),
                        flags=cv2.INTER_LINEAR)
    return warped
