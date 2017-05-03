# Advanced Lane Detection
### May 3rd, 2017
## By Satchel Grant

---

**Advanced Lane Finding Project**

This was a project completed for Udacity's Self Driving Car Engineer nano degree.

The goals / steps of this project were the following:

* Compute a camera calibration matrix and distortion coefficients given using a set of chessboard images.
* Apply a distortion correction to raw images.
* Apply a perspective transform to road images ("birds-eye view").
* Use color transforms, gradients, etc., to create a thresholded binary image highlighting lane lines.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Transform the detected lane boundaries back onto the original image.
* Display calculated lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted.jpg "Undistorted"
[image1b]: ./output_images/undistorted_road.jpg "Undistorted Road"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/saturated.jpg "Saturated Example"
[image3b]: ./output_images/filtered.jpg "Filtered Example"
[image4]: ./output_images/birds-eye.jpg "Warp Example"
[image5]: ./output_images/filled.jpg "Fit Visual"
[image6]: ./output_images/overlayed.jpg "Output"
[video1]: ./processed_project_video.mp4 "Video"


### This README addresses each point on the [project rubric](https://review.udacity.com/#!/rubrics/571/view)

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is split between the [calibration](./calibration.py) python file and the 3rd and 4th cells of the [IPython notebook](./advanced_lane_finder.ipynb).  

I started by preparing "object points", which are the (x, y, z) coordinates of the chessboard corners in space. I assumed the chessboard was fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.

Referring to the code, `objp` is just a replicated array of coordinates, and `objpoints` were appended with a copy of `objp` every time the chessboard corners were successfully detected in a test image.  `imgpoints` are a list of the (x, y) pixel positions of each of the corners in the image plane from each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. These coefficients could then be used with the `cv2.undistort()` function to undistort images.

![alt text][image1]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The following is an image with distortion correction. The correction is not very noticeable, but there all the same.
![alt text][image1b]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I first converted the images to HLS and isolated the saturation because this gave the best lane image.

Converted to HLS and take only saturation.
![alt text][image3]

I then used sobel gradient detection in the x direction, a combined sobel gradient in multiple directions, and the original saturation pixels to generate a binary image (thresholding steps at lines 83 through 92 in `filters.py`).  Here's an example of my output for this step.

Multiple gradient thresholding techniques.
![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `change_persp()`, which is defined in lines 127 through 130 in the file `calibration.py`. It is used for the first time in the 5th code cell of the [IPython](./advanced_lane_finder.ipynb) notebook. The `change_persp()` function takes in an image and a transform matrix and returns a bird's eye view of the lane.

The transform matrix used in `change_persp()` is calculated from the function `get_transform` defined in lines 104-119 in `calibration.py`. I chose the hardcode the source points for calculating the transform matrix, but the destination points are based off of the optional parameter `offset`. They are defined in the following manner:

```python
src_pts = np.float32([[277,680],[555,475],[727,474],[1049,680]])

offset=(250,0)
top_offset=20
dst_pts = np.float32([[xoffset, img_size[0]-yoffset],
                      [xoffset, yoffset+top_offset],
                      [img_size[1]-xoffset, yoffset+top_offset],
                      [img_size[1]-xoffset, img_size[0]-yoffset]])
```

Here is an example of the perspective transform:

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The first step to fitting a polynomial to the lane lines was to isolate the most likely location for the base of each lane line. I did this by counting the non-zero pixels in each column of the image up to halfway up the image. The columns with largest counts had the greatest chance of being the columns centered on the lane lines. See `find_start_cols()` defined in lines 29-40 in [lanes](./lanes.py) for this step.

The next step was to create boxes (or windows) for each lane and slide them up along the lane's main column picked by the histogram. The pixels within the windowed regions were then kept and used to fit a polynomial to the lanes (uses functions `find_lane_pixels()` and `fit_poly()` defined in lines 42-233 in [lanes](./lanes.py)). The space between the two fitted polynomials was then filled with color and transformed back to its original orientation using `fill_lane()` at lines 209-227 in `lanes.py` followed by `change_persp()` in calibration.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Calculating the radius of curvature of the lane lines involved some math:

R = (1 + ( f' )<sup>2</sup> )<sup>3/2</sup> / f''

Where R is the radius of curvature and f is the lane line's fitted polynomial. f' and f'' denote the first and second derivative respectively. The polynomial fits are 2nd order which makes the derivatives relatively easy to get. Given f as the polynomial:

f = ø<sub>1</sub>x<sup>2</sup> + ø<sub>2</sub>x + ø<sub>3</sub>

f' = 2ø<sub>1</sub>x + ø<sub>2</sub>

f'' = 2ø<sub>1</sub>

Where each ø is a fitted parameter for the polynomial. At this point, it was easy to plug and chug to find the radial curvature in terms of pixels. The units of measurement were then converted from pixels to meters. The conversion for the lane is found from the actual width of the lane in the US of 3.7m divided by the width of the lane in pixels, 775px. See [curve_radius](./lanes.py) (lines 180-196 in `lanes.py`) for implementation details.

The distance of the lanes from the car were found by assuming the car is located in the center of the image and finding the average of the difference in pixel location of the fitted lane lines from the center of the image. The result was then converted to meters from pixels. See `car_location()` defined at lines 198-206 in `lanes.py`.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in code cell 8 in `advanced_lane_finder.ipynb` using the functions `fill_lane()`, `change_persp()`, and `overlay()` (found in lines 260-268 in `lanes.py`).  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

At one point on my first run through on the project video, the right lane line polynomial would drift over to the left lane and stay there. In order to fix this, I would recalculate the lane line polynomials from scratch if the update polynomials ever drifted across the mid line of the image (see lines 125-130 of `lanes.py`). This fixed the problem.

I experienced problems with calculating the curvature of the lanes with a good degree of precision. The curvature for the right vs left lanes often had drastically different values. This is obviously incorrect since the lanes are often parallel and relatively straight. The lack of precision could be improved by including a running average of past curvature measurements in the calculations. It may also be helpful to combine and take the average of the two separate lane curvature measurements since the difference between them should only ever be about 3.7m. Both of these ideas could be further improved by measuring the confidence of the fitted lane lines and giving greater weight in the averages to high confidence measurements. The confidence could be measured by pixel counts when finding the lane lines.

The challenge videos went poorly. They could be improved by measuring confidence of polynomial fit forcing conformity to more confident measurements. In the harder video, the center lane line is more consistent than the right which could be used to base most measurements off. The lane fitting could be improved by enforcing the lane distance between the two fitted polynomials. This would ensure the lane width remains constant. Fitting could also perhaps further be improved with more tuned filter parameters.

An interesting approach would be to use a convolutional neural net to fit the lanes. I believe it would be doable but could require a significant amount of hand labeling. It would likely require a bit more data, too. Although data augmentation goes a long way.
