

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
            img = cv.imread(os.path.join(folder, filename),cv.CV_16U)
            if img is not None:
                images.append(img)
    return images


