# Importing necessary library
import numpy as np
import cv2 as cv
# from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import SupportFunctions as mySF
from sklearn.cluster import KMeans


TOP_IMAGES_PATH = "./camera_top"

images = mySF.load_images_from_folder(TOP_IMAGES_PATH)

#   We will proceed with the first image as the leading example. 
photo = images[0]

blue_bricks = cv.imread('./lena.png')
show_bricks = cv.cvtColor(blue_bricks, cv.COLOR_BGR2GRAY)
#dst = cv.calcHist(images[0], [0], None, [255], [0,255])

arr_flatten = photo.flatten() 
print(photo.shape)

vmax =max(arr_flatten)
vmin =min(arr_flatten)
print(vmax)
print(vmin)


normalized = np.zeros(photo.shape)
plt.figure(1)
plt.imshow(cv.convertScaleAbs(photo,alpha=(255.0/65535.0)),cmap='gray')
plt.figure(2)
normalized = cv.normalize(photo,None,0,65535,cv.NORM_MINMAX)
normalized = cv.convertScaleAbs(normalized,alpha=(255.0/65535.0))
# normalized = cv.normalize(photo,None,0,255,cv.NORM_MINMAX, dtype = normalized.dtype)
print(normalized.dtype)
plt.imshow(normalized)


#hist_values = cv.calcHist(show_bricks,channels=[0],mask=None,histSize=[256],ranges=[0,256])

arr_flatten = normalized.flatten() 
vmax =max(arr_flatten)
vmin =min(arr_flatten)
print(vmax)
print(vmin)


plt.figure(3)
plt.imshow(normalized,cmap='gray')

plt.figure(4)
hist_values = cv.calcHist([normalized],channels=[0],mask=None,histSize=[256],ranges=[0,255])
plt.plot(hist_values)




# kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0).fit(df)
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=0, max_iter=1)
kmeans.fit(arr_flatten.reshape(-1,1))  # at least 2 dimensions, so we just set everything into the same space

plt.show()


# plt.figure(1, figsize=(15, 5))
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))


# plt.subplot(221)
# plt.imshow(photo,label='Original16b')

# plt.subplot(222)

# bin_counts, bin_edges = np.histogram(photo, bins)
# plt.hist(dstO, bins0, color='blue', label='data')
# # plt.plot(x1, bins, color='blue', marker='o', label='data')
# # plt.plot(x2, damp(x2), color='black', label='datapoints')

# plt.subplot(223)
# plt.imshow(cvuint8,label='Converted8b')

# plt.subplot(224)
# hist_values = cv.calcHist(cvuint8,channels=[0],mask=None,histSize=[256],ranges=[0,256])
# plt.plot(hist_values)
# bin_counts, bin_edges = np.histogram(cvuint8, bins)
# plt.hist(dst, bins, color='blue', label='data')

plt.show()

plt.show()