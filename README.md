# **Vehicle Detection Project**

My project includes the following files:
* pipeline.ipynb - Jupyter notebook to process the original video
* lesson_functions.py - helper functions used by the pipeline. Most of them are from the lession.
* video.mp4 - output video with vehicles marked with yellow boxes
* README.md - summarizing the results
* svc_c100.pkl - trained SVM model

The goals / steps of this project are the following:

* Extract features from the spatial binary, color histogram, histogram of oriented gradients (HOG) on a labeled training set of images and train a SVM classifier.
* Implement a sliding-window technique to search for vehicles in images.
* Build a pipeline to process the video stream to detect vehicles frame by frame.
* Estimate a bounding box for vehicles detected.

---
### Feature extraction

I use the training set provided by the [project](https://github.com/udacity/CarND-Vehicle-Detection) ([vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)) to train my SVM classifier. I choose `sklearn.svm.svc` with RBF kernel because the test score is better than the one fitted with linear kernel `sklearn.svm.LinearSVC`. I extract spatial binary, color histogram, and HOG features using `extract_features` function defined in `lesson_functions.py`. The combined features are normalized by `sklearn.preprocessing.StandardScaler`. The whole training process is described in cell #2 in my notebook, which will produce a trained SVM classifier.  

Here are examples of HOG features in `Y`, `Cr` and `Cb` channels with (`orientations`, `pixels_per_cell`, `cells_per_block`) set to (9, 8, 1).
![alt text](./output_images/hog.png "HOG")
I choose `YCrCb` color space because it preserves the color under varying illumination conditions ([this](http://www.learnopencv.com/color-spaces-in-opencv-cpp-python/) is a good article to read). I also tried `HSL` color space, the one I used in [Project 4](https://github.com/enhsin/p4-advancedLaneLines) to dectect lane lines. It seems to give more false negatives. 

Channel `Y` appear 
`orientations`, `pixels_per_cell`, and `cells_per_block`

#### Histogram of Oriented Gradients (HOG)


####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

#### spatial binary, color histogram, and HOG features
![alt text](./output_images/features.png "features")


### Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text](./output_images/window.png "window")

![alt text](./output_images/window_small.png "window_small")

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

![alt text](./output_images/heat.png "heat")



---

### Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

