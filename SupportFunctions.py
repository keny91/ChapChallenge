

# These are a set of support functions to assist 

import numpy as np
import cv2 as cv
# from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import os

from pandas import wide_to_long



###     Loads all the images in a folder valid 'jpg' and 'tif'
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".tif") :
            img = cv.imread(os.path.join(folder, filename), -1)
            if img is not None:
                images.append(img)
    return images


###     Get NofImages at folder
def get_nof_images_at_folder(folder):
    nof = 0
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".tif") :
            nof += 1
    return nof

###     Get image af folder by index. Index starts at 0
#   get_nof_images_at_folder should be done beforeHand to avoid entering an invalid index
def load_image_from_folder_byIndex(folder,index):
    i = 0;
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".tif") :
            if index == i :
                return cv.imread(os.path.join(folder, filename), -1)
            else : 
                i +=1

    return None

def getImageCutoffAroundPoint(image,point,distance):
    
    # missing validation for out of bounds... skippping because the search areas are very central
    # if()
    x = point[0]
    y = point[1]
    crop_img = image[y-distance:y+distance, x-distance:x+distance]

    return  crop_img 
        
def showQuickImage(image,tittle):
    fig = plt.figure(1)
    fig, (ax1) = plt.subplots(1, 1)
    ax1.set_title(tittle)
    ax1.imshow(image, cmap="gray")    


def find_board( image):
        """
        Finds a board by calling openCV function to find contures in image.
        Than it sorts those contures and stores the biggest one.
        In case there is more than one we go over all found contures and
        keep only one with 4 points

        Args:
            image(numpy.ndarray): Image to find contures from

        Returns:
            Found conture in given image
        """
        
        
        im = image.copy()
        width = image.shape[1]
        height = image.shape[0]
        im = cv.dilate(im, ((5, 5)), iterations=8)
        (cnts, _) = cv.findContours(im,
                                    cv.RETR_TREE,
                                    cv.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:10]
        our_cnt = None
        for c in cnts:
            peri = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, 0.1 * peri, True)
            if len(approx) == 4:
                # The board needs to be at least 1/3 of the image size
                min_size = np.array([width * 1/10.0, height * 1/15.0])

                a = np.abs(approx[0] - approx[2])[0] > min_size
                b = np.abs(approx[1] - approx[3])[0] > min_size
                true = [True, True]
                if np.array_equal(a, true) or np.array_equal(b, true):
                    our_cnt = approx
                break

        return our_cnt

#dilation
def dilate(image, units):
    kernel = np.ones((units,units),np.uint8)
    return cv.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image,units):
    kernel = np.ones((units,units),np.uint8)
    return cv.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image,units):
    kernel = np.ones((units,units),np.uint8)
    return cv.morphologyEx(image, cv.MORPH_OPEN, kernel)

def close(image,units):
    kernel = np.ones((units,units),np.uint8)
    return cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
    