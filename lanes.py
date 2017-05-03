import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import math


def window_topbottom(img_shape, n, height):
    bottom = img_shape[0] - (n+1) * height
    top = img_shape[0] - n * height
    return bottom, top

def window_walls(current_column, margin):
    left_wall = current_column - margin
    right_wall = current_column + margin
    return left_wall, right_wall

def find_start_cols(filt_img):
    mid_col = filt_img.shape[1]//2
    search_region = filt_img.shape[0]//2
    # Use histogram on image to find lane base locations
    hist = np.sum(filt_img[search_region:,:],axis=0)
    left_cols_base = np.argmax(hist[:mid_col])
    right_cols_base = np.argmax(hist[mid_col:]) + mid_col
    return left_cols_base, right_cols_base

def find_lane_pixels(filt_img,n_windows=10,margin=100,minpixels=50):
    filt_img[filt_img != 0] = 1
    output = np.dstack([filt_img*255 for i in range(3)])
    midpt = output.shape[1]//2

    # Use histogram on image to find base locations of lanes
    left_cols_base, right_cols_base = find_start_cols(filt_img)

    # Make windows to isolate potential lane pixels
    window_height = output.shape[0]//n_windows
    nonzeros = filt_img.nonzero()
    nz_rows, nz_cols = nonzeros[0], nonzeros[1]

    left_current_column = left_cols_base
    right_current_column = right_cols_base

    left_lane_indices = []
    right_lane_indices = []

    for window in range(n_windows):
        # Find window edges
        win_bottom, win_top = window_topbottom(filt_img.shape,window,window_height)

        left_lwall, left_rwall = window_walls(left_current_column, margin)
        right_lwall, right_rwall = window_walls(right_current_column, margin)

        # Draw the windows on the output
        cv2.rectangle(output,(left_lwall,win_bottom),
                            (left_rwall,win_top),
                            (0,255,0), 2)
        cv2.rectangle(output,(right_lwall,win_bottom),
                            (right_rwall,win_top),
                            (0,255,0), 2)

        # Get indices of the nonzero pixels in within the window
        nz_left_indices = ((nz_rows >= win_bottom) &
                            (nz_rows < win_top) &
                            (nz_cols >= left_lwall) &
                            (nz_cols < left_rwall)).nonzero()[0]
        nz_right_indices = ((nz_rows >= win_bottom) &
                            (nz_rows < win_top) &
                            (nz_cols >= right_lwall) &
                            (nz_cols < right_rwall)).nonzero()[0]

        # Append the nonzero indices to the index lists
        left_lane_indices.append(nz_left_indices)
        right_lane_indices.append(nz_right_indices)

        # If more than minpixels are within a window, window recenters on mean
        if len(nz_left_indices) > minpixels:
            left_current_column = np.int(np.mean(nz_cols[nz_left_indices]))
        if len(nz_right_indices) > minpixels:
            right_current_column = np.int(np.mean(nz_cols[nz_right_indices]))

    # Concatenate respective lane indices together into individual arrays
    left_lane_indices = np.concatenate(left_lane_indices)
    right_lane_indices = np.concatenate(right_lane_indices)

    # Extract left and right lane pixel positions
    l_nzlane_rowcols = nz_rows[left_lane_indices],\
                            nz_cols[left_lane_indices]
    r_nzlane_rowcols = nz_rows[right_lane_indices],\
                            nz_cols[right_lane_indices]

    return l_nzlane_rowcols, r_nzlane_rowcols

def poly_params(nzlane_rowcols,degree=2):
    rows, cols = nzlane_rowcols
    fit = np.polyfit(rows, cols, degree)
    return fit

def fit_poly(img_shape, nzlane_rowcols):
    params = poly_params(nzlane_rowcols)
    plot_rows = np.linspace(0, img_shape[0]-1, img_shape[0])
    fit_cols = params[0]*plot_rows**2 + params[1]*plot_rows + params[2]
    return plot_rows, fit_cols, params

def update_nzindices(params, nz_cols, nz_rows, margin):
    return ((nz_cols > (params[0]*nz_rows**2 +\
                        params[1]*nz_rows + params[2] - margin)) &\
                        (nz_cols < (params[0]*nz_rows**2 +\
                        params[1]*nz_rows + params[2] + margin)))

def find_lanes_update(filt_img, l_nzlane_rowcols, r_nzlane_rowcols, margin=75):
    left_fit = poly_params(l_nzlane_rowcols)
    right_fit = poly_params(r_nzlane_rowcols)
    midpt = filt_img.shape[1]//2

    nonzeros = filt_img.nonzero()
    nz_rows, nz_cols = nonzeros[0], nonzeros[1]

    left_lane_indices = update_nzindices(left_fit, nz_cols, nz_rows, margin)
    right_lane_indices = update_nzindices(right_fit, nz_cols, nz_rows, margin)

    l_nzlane_rowcols = nz_rows[left_lane_indices],\
                            nz_cols[left_lane_indices]
    # Insurance that update goes smoothly
    if np.max(l_nzlane_rowcols[1]) > midpt:
        return find_lane_pixels(filt_img)
    r_nzlane_rowcols = nz_rows[right_lane_indices],\
                            nz_cols[right_lane_indices]
    # Insurance that update goes smoothly
    if np.min(r_nzlane_rowcols[1]) < midpt:
        return find_lane_pixels(filt_img)

    return l_nzlane_rowcols, r_nzlane_rowcols


def curve_radius(nzlane_rowcols, row_mpp=30/720,
                                        col_mpp=3.7/775):
    eval_row = np.max(nzlane_rowcols[0])*row_mpp
    nzlane_rowcols = (nzlane_rowcols[0]*row_mpp,
                        nzlane_rowcols[1]*col_mpp)

    fit_params = poly_params(nzlane_rowcols)

    radius = ((1 + (2*fit_params[0]*eval_row +
                    fit_params[1])**2)**1.5)/abs(2*fit_params[0])
    base_location = fit_params[0]*eval_row + fit_params[1]*eval_row + fit_params[2]
    return radius, base_location

def car_location(lane_loc, midpt, col_mpp=3.7/775):
    car_offset = abs(col_mpp*midpt-lane_loc)
    return car_offset


def fill_lane(img, l_nzlane_rowcols, r_nzlane_rowcols, margin=5, params=False):
    copy = img.copy()
    plot_row, left_fit_col, lparams = fit_poly(img.shape,l_nzlane_rowcols)
    plot_row, right_fit_col, rparams = fit_poly(img.shape,r_nzlane_rowcols)
    left_fit_col = left_fit_col.astype('int32')
    right_fit_col = right_fit_col.astype('int32')
    for i in range(len(img)):
        copy[i,left_fit_col[i].astype('int32'):right_fit_col[i].astype('int32')] = [0,255,100]
        copy[i,left_fit_col[i]-margin:left_fit_col[i]+margin] = [0,0,255]
        copy[i,right_fit_col[i]-margin:right_fit_col[i]+margin] = [0,0,255]
    if params: return copy, (lparams, rparams)
    return copy

def draw_fitted_lanes(img, l_nzlane_rowcols, r_nzlane_rowcols, margin=5, params=False):
    copy = img.copy()
    plot_row, left_fit_col, _ = fit_poly(img.shape,l_nzlane_rowcols)
    plot_row, right_fit_col, _ = fit_poly(img.shape,r_nzlane_rowcols)
    left_fit_col = left_fit_col.astype('int32')
    right_fit_col = right_fit_col.astype('int32')
    for i in range(len(img)):
        copy[i,left_fit_col[i]-margin:left_fit_col[i]+margin] = [0,0,255]
        copy[i,right_fit_col[i]-margin:right_fit_col[i]+margin] = [0,0,255]
    if params: return copy, (lparams, rparams)
    return copy

def highlight_lanes(l_nzlane_rowcols, r_nzlane_rowcols):
    left_rows, left_cols = l_nzlane_rowcols
    right_rows, right_cols = r_nzlane_rowcols
    img[left_rows, left_cols] = [255, 0, 0]
    img[right_rows, right_cols] = [0, 0, 255]
    return img

def overlay(img1, img2, portion1=.5, portion2=.5):
    return cv2.addWeighted(img1, portion1, img2, portion2, 0)
