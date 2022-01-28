# Importing necessary library
import numpy as np
import cv2 as cv
# from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import SupportFunctions as mySF


TOP_IMAGES_PATH = "./camera_top"
images = mySF.load_images_from_folder(TOP_IMAGES_PATH)
photo=images[0]
#lET US SEE THE SHAPE OF THE IMAGE.

min_val,max_val,min_indx,max_indx=cv.minMaxLoc(photo)
plt.imshow(photo)
# map.convertTo(photo,cv.CV_8UC1, 255 / (max-min), -min);
cvuint8 = cv.convertScaleAbs(photo,alpha=(255.0/65535.0))

cv.imshow("Out", cvuint8)
# plt.imshow(cvuint8)
min_val,max_val,min_indx,max_indx=cv.minMaxLoc(cvuint8)


