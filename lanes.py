import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import math


def partition_lanes(filtered_img, h_ratio=40/64, w_frac=13, car_hood=0, fill_color=(255,0,0)):
    ysize, xsize = filtered_img.shape
    poly_left = np.zeros_like(filtered_img)
    poly_right = np.zeros_like(filtered_img)

    polyVerticesLeft = np.array([[(0, ysize-car_hood),
                                (np.floor(w_frac/2)*xsize/w_frac, h_ratio*ysize),
                                (xsize/2, h_ratio*ysize),
                                (np.floor(w_frac/2)*xsize/w_frac, ysize-car_hood)]],
                                dtype=np.int32)
    polyVerticesRight = np.array([[(np.ceil(w_frac/2)*xsize/w_frac,ysize-car_hood),
                                    (xsize/2, h_ratio*ysize),
                                    (np.ceil(w_frac/2)*xsize/w_frac, h_ratio*ysize),
                                    (xsize, ysize-car_hood)]],
                                    dtype=np.int32)

    cv2.fillPoly(poly_left, polyVerticesLeft, fill_color)
    cv2.fillPoly(poly_right, polyVerticesRight, fill_color)

    poly_left = np.logical_and(poly_left,filtered_img)
    poly_right = np.logical_and(poly_right,filtered_img)
    return poly_left, poly_right


def draw_lines(img,lines,fill_color=(255,0,0),width=2):
    for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), fill_color, width)
    return img

def avg_vector(lines):
    vec = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            dist = math.sqrt((x2-x1)**2+(y2-y1)**2)
            vec.append([(x2-x1)*dist,(y2-y1)*dist])
    return np.sum(vec,axis=0)/len(vec)

def ptvecs(lines):
    pt = (np.sum(lines,axis=0)/len(lines))[0]
    pt = [(pt[0]+pt[2])/2,(pt[1]+pt[3])/2]
    vec = avg_vector(lines)
    return (pt,vec)

def get_line(ptvec,img_size,h_ratio=40/64):
    avg_pt = ptvec[0]
    avg_vec = ptvec[1]
    slope = avg_vec[1]/avg_vec[0]
    y1 = img_size[0]
    y2 = h_ratio*img_size[0]
    x1 = round((y1-avg_pt[1])/slope+avg_pt[0])
    x2 = round((y2-avg_pt[1])/slope+avg_pt[0])
    return (int(x1),int(y1),int(x2),int(y2))

def get_lane_lines(filtered_img):
    filt_left,filt_right = partition_lanes(filtered_img)
    left_lines = cv2.HoughLinesP(filt_left.astype(np.uint8),
                                5,
                                np.pi/180,
                                20,
                                np.array([]))
    right_lines = cv2.HoughLinesP(filt_right.astype(np.uint8),
                                5,
                                np.pi/180,
                                20,
                                np.array([]))
    leftptvec = ptvecs(left_lines)
    rightptvec = ptvecs(right_lines)
    left_line = get_line(leftptvec,filtered_img.shape)
    right_line = get_line(rightptvec,filtered_img.shape)
    return left_line,right_line

def draw_lanes(filtered_img):
    left_line,right_line = get_lane_lines(filtered_img)
    return draw_lines(np.zeros_like(filtered_img),[[left_line],[right_line]])
