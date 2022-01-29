

# These are a set of support functions to assist 

import numpy as np
import cv2 as cv
# from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import os



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

