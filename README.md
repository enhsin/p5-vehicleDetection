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

I use the training set provided by the [project](https://github.com/udacity/CarND-Vehicle-Detection) ([vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)) to train my SVM classifier. I choose `sklearn.svm.svc()` with RBF kernel because the test score is better than the one fitted with linear kernel `sklearn.svm.LinearSVC()`. I extract spatial binary, color histogram, and HOG features using `extract_features` function defined in `lesson_functions.py`. The combined features are normalized by `sklearn.preprocessing.StandardScaler()`. The whole training process is described in cell #2 in my notebook, which will produce a trained SVM classifier.  

Here are examples of HOG features in `Y`, `Cr` and `Cb` channels with (`orientation`, `pix_per_cell`, `cell_per_block`) set to (9, 8, 1).
![alt text](./output_images/hog.png "HOG")
I choose `YCrCb` color space because it preserves the color under varying illumination conditions ([this](http://www.learnopencv.com/color-spaces-in-opencv-cpp-python/) is a good article to read). I also tried `HSL` color space, which was used in [Project 4](https://github.com/enhsin/p4-advancedLaneLines) to dectect lane lines. It seems to give more false negatives when processing the video stream. 

Channel `Cr` doesn't appear to be useful to detect the shape of the object, so I only use Channel `Y` for the HOG feature in the final pipeline.

Here is the visualization of all the feature vectors with my final choice of parameters.
![alt text](./output_images/features.png "features")

Here are the final parameters.

|Spatial Parameter| Value |
|-----------------|-------|
|channel          |Y,Cr,Cb|
|spatial_size     |8      |

spatial feature size = 8x8x3 = 192

|Color Parameter  | Value |
|-----------------|-------|
|channel          |Cr,Cb  |
|hist_bins        |16     |

color feature size = 16x2 = 32

|HOG Parameter    | Value |
|-----------------|-------|
|channel          |Y      |
|orientation      |4      |
|pix_per_cell     |8      | 
|cell_per_block   |1      | 

HOG feature size = 4x(64/8)x(64/8) = 256

Total feature size = 480

`pix_per_cell` is set to 8 because the image in the training set 64x64 in size and I want all the pixels to be used. This will give 64 cells. `cell_per_block` is fixed at 1 because I prefer a smaller set of features to reduce the prediction time. I loop through various combinations of `orientation`, `hist_bins`, and `spatial_size` (cell #3). They all give an accuracy about 0.99. I select (`orientation`, `hist_bins`, `spatial_size`) = (4, 16, 8) because of the speed and performance. Some models will take 2-3 hours to process the whole video and the one I choose takes 23 minutes on the virtual machine of my windows laptop.  


### Sliding Window Search

I use the method taught in Session 35 (Hog Sub-sampling Window Search) to search for cars. The code is described in `lesson_functions.find_cars()`. `cells_per_step` is set to 2 (75% of overlapping) to increase the chance of detection and to use multiple detections to reject false positives. The advantage of this method is to extract HOG features once and to avoid the expensive multiple HOG calculations. I run this function at two scales (0.85, 1.8) to detect both near and far objects. A mask (defined in cell #6) is also used to focus on the right two lanes.

Here are examples of the detection (blue boxes). There are two kinds of blue boxes, big and small (they are overlapping). The yellow boxes are the final detection. 

![alt text](./output_images/window.png "window") ![alt text](./output_images/window_small.png "window_small")

There are more examples of the detection shown in cell #8.

### Multiple detections

I apply the method in Session 37 (Multiple Detections & False Positives) to combine multiple detections to produce the bounding box of the detected vehicle. The code is at `lesson_functions.threshold_boxes()`. I create a heatmap from multiple detections of a single frame and use `scipy.ndimage.measurements.label()` to label connected components. A slight modification to the code used in the lesson is that I use two thresholds, `thresh_hi`  and `thresh_lo`. The maximum count of a connected component (number of detection) must be equal or greater than `thresh_hi` to be considered as a real detection. The heatmap pixels with counts greater than `thresh_lo` are used to mark the final bounding box. I set `thresh_hi`= 4 and `thresh_lo` = 1. 

Here are the images of the raw input data (left), the heatmap of the detection (middle), and the final processed image (right). The bounding box is drawn with thick lines. The blue boxes are sliding window detections.

![alt text](./output_images/heat.png "heat")

The input image has been cropped because that is the region of interest. I flip the image so that features can be extracted starting from the lower right corner instead of the upper left. If the image is not flipped, there could be pixels on the right and on the bottom unanalyzed (depending on `scales`, `pix_per_cell`, `cell_per_block` or `cells_per_step`) and we want to detect cars entering into the frame early. 

### Video Implementation

The pipeline to process each frame of the project video is described in cell #7. Here is the output [video](./video.mp4). The yellow box is the final detection and blue boxes are sliding window detections. False positives are rejected by setting the proper `thresh_hi` (see above). I also set `class_weight` = {0:0.9, 1:0.1} (in `sklearn.svm.svc()`) to give non-vehicles more weights and to change the decision boundary a little bit.

---

### Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

