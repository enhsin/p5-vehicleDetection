import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
from scipy.ndimage.measurements import label

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, color_space, nbins=32):
    if color_space == 'HLS':
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=(0.,360.))
        hist_features = channel1_hist[0]
    elif color_space == 'YCrCb':
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=(0.,1.))
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=(0.,1.))
        hist_features = np.concatenate((channel2_hist[0], channel3_hist[0]))
    else:
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=(0.,1.))
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=(0.,1.))
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=(0.,1.))
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True,
                        flip=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        if flip:
            image = image[::-1,::-1]
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, color_space, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            elif hog_channel == '12':
                hog_features = []
                for channel in [1,2]:
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features
    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, color_space, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
        elif hog_channel == '12':
            hog_features = []
            for channel in [1,2]:
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, mask, boxes, scales, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    on_windows = []
    found = np.zeros_like(mask).astype(np.int)
    for i in range(len(scales)):
        box = boxes[i]
        scale = scales[i]
        nx = np.int((box[1][0] - box[0][0])/scale)
        ny = np.int((box[1][1] - box[0][1])/scale)
        img_tosearch = cv2.resize(img[box[0][1]:box[1][1], box[0][0]:box[1][0]], (nx, ny))

        if color_space == 'HSV':
            img_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            img_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            img_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            img_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            img_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)

        # Define blocks and steps as above
        nxblocks = (img_tosearch.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (img_tosearch.shape[0] // pix_per_cell) - cell_per_block + 1

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

        # Compute individual channel HOG features for the entire image
        if hog_channel == 'ALL' or hog_channel == 0:
            hog0 = get_hog_features(img_tosearch[:,:,0], orient, pix_per_cell, cell_per_block, feature_vec=False)
        if hog_channel == '12' or hog_channel == 1 or hog_channel == 'ALL':
            hog1 = get_hog_features(img_tosearch[:,:,1], orient, pix_per_cell, cell_per_block, feature_vec=False)
        if hog_channel == '12' or hog_channel == 2 or hog_channel == 'ALL':
            hog2 = get_hog_features(img_tosearch[:,:,2], orient, pix_per_cell, cell_per_block, feature_vec=False)

        for xb in range(nxsteps):
            xpos = xb*cells_per_step
            xpos1 = min(xpos+nblocks_per_window,img_tosearch.shape[1])
            xleft = xpos*pix_per_cell
            xright = xpos1*pix_per_cell
            xbox_left = np.int(xleft*scale + box[0][0])
            xbox_right = np.int(xright*scale + box[0][0])
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                ypos1 = min(ypos+nblocks_per_window,img_tosearch.shape[0])
                ytop = ypos*pix_per_cell
                ybottom = ypos1*pix_per_cell
                ybox_top = np.int(ytop*scale + box[0][1])
                ybox_bottom = np.int(ybottom*scale + box[0][1])
                # if the cell is outside of the region of interest
                if mask[(ybox_top+ybox_bottom)//2,(xbox_left+xbox_right)//2] == 0:
                    continue

                img_features = []
                feature_image = cv2.resize(img_tosearch[ytop:ybottom, xleft:xright], (64,64))
                if spatial_feat == True:
                    spatial_features = bin_spatial(feature_image, size=spatial_size)
                    img_features.append(spatial_features)
                if hist_feat == True:
                    hist_features = color_hist(feature_image, color_space, nbins=hist_bins)
                    img_features.append(hist_features)
                if hog_feat == True:
                    # Extract HOG for this patch
                    if hog_channel == 'ALL':
                        hog_feat0 = hog0[ypos:ypos1, xpos:xpos1].ravel()
                        hog_feat1 = hog1[ypos:ypos1, xpos:xpos1].ravel()
                        hog_feat2 = hog2[ypos:ypos1, xpos:xpos1].ravel()
                        hog_features = np.hstack((hog_feat0, hog_feat1, hog_feat2))
                    elif hog_channel == '12':
                        hog_feat1 = hog1[ypos:ypos1, xpos:xpos1].ravel()
                        hog_feat2 = hog2[ypos:ypos1, xpos:xpos1].ravel()
                        hog_features = np.hstack((hog_feat1, hog_feat2))
                    elif hog_channel == 0:
                        hog_features = hog0[ypos:ypos1, xpos:xpos1].ravel()
                    elif hog_channel == 1:
                        hog_features = hog1[ypos:ypos1, xpos:xpos1].ravel()
                    elif hog_channel == 2:
                        hog_features = hog2[ypos:ypos1, xpos:xpos1].ravel()
                    img_features.append(hog_features)
                test_features = scaler.transform(np.concatenate(img_features).reshape(1,-1))
                test_prediction = clf.predict(test_features)
                if test_prediction == 1:
                    on_windows.append(((xbox_left, ybox_top),(xbox_right,ybox_bottom)))

    return on_windows

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def transform_single(w,x_size,y_size):
    return ((x_size-w[1][0],y_size-w[1][1]),(x_size-w[0][0],y_size-w[0][1])
)
def transform(windows,x_size,y_size):
    return [transform_single(w,x_size,y_size) for w in windows]

def threshold_boxes(draw_img, img, hot_windows, thresh_hi, thresh_lo, x, y, color=(0, 0, 255), thick=6, vis=False):
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    heat = add_heat(heat,hot_windows)
    if vis:
        heatmap = np.copy(heat)
        label_img = np.copy(img)
    heat[heat<=thresh_lo] = 0
    labels = label(heat)
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        max_count = np.max(heat[nonzero])
        if max_count < thresh_hi:
            continue
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        if vis:
            cv2.rectangle(heatmap, bbox[0], bbox[1], np.max(heatmap), thick)
            cv2.rectangle(label_img, bbox[0], bbox[1], color, thick)
        bbox = transform_single(bbox,x,y)
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    if not vis:
        return draw_img
    else:
        return draw_img, heatmap, label_img


