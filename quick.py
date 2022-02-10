

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import SupportFunctions as mySF
from sklearn.cluster import KMeans



def Convert8bAndNormalize ( image16b ):
    normalized = cv.normalize(image16b,None,0,65535,cv.NORM_MINMAX)
    return cv.convertScaleAbs(normalized,alpha=(255.0/65535.0))
    
def Convert8b (image16b):
    max_val = image16b.max()
    # return cv.convertScaleAbs(image16b,alpha=(255.0/max_val))
    return cv.convertScaleAbs(image16b,alpha=(255.0/65535.0))
    
def Normalize16b ( image16b ):
    return cv.normalize(image16b,None,0,65535,cv.NORM_MINMAX)


# will return 3 values indicating the depth of [bg, middleboards,topboards]
def getDepthLayersCenters(images, nof_clusters = 3):
    
    arr_ = []
    # cluster model
    
    scale_percent = 30
    #kmeans = blobs.KMeans(n_clusters=3, init='k-means++', random_state=0, max_iter=1)
    for ima in images:
        image8bNorm = Convert8bAndNormalize ( ima ) 
        width = int(image8bNorm.shape[1] * scale_percent / 100)
        height = int(image8bNorm.shape[0] * scale_percent / 100)
        dim = (width, height)
  
        # resize image to a 30% the original slcale
        resized = cv.resize(image8bNorm, dim)
        
        # validate not null - skipped
        arr_flatten = resized.flatten()
        kmeans = KMeans(n_clusters=3, init='k-means++', random_state=0, max_iter=1)
        kmeans.fit(arr_flatten.reshape(-1,1)) 
        arr_.append(sorted(np.array(kmeans.cluster_centers_)))
            
    # convert from list to array    
    arr_ = np.array(arr_)
    
    nof_images = np.array(arr_).shape[0]
    nof_centers = np.array(arr_).shape[1]
    
    avr_centers = np.sum(arr_,axis=0)/nof_images

    # round and return
    return np.rint(sorted(avr_centers))


def median_value(val1, val2):
    threshold = np.rint((val1+val2)/2)
    threshold = int(threshold)
    return threshold

def getLayerByGreyRange(image, minTH, maxTH):
    
    #  th, im_th = cv.threshold(image, minTH, maxTH, cv.THRESH_BINARY)
    im_th = cv.inRange(image, minTH, maxTH)

    return im_th


def getHigherLayer(image8bNorm_, cluster_c, verbose_ = False):
    
    # top boards layer
    # threshold = th_value
    threshold = median_value(cluster_c[len(cluster_c)-2],cluster_c[len(cluster_c)-1])
    max_threshold = 255 # till whitest white

    hist_values_norm = cv.calcHist([image8bNorm_.flatten()],channels=[0],mask=None,histSize=[256],ranges=[0,255])    

    # Get binarized image of the layer
    imageTopTH = getLayerByGreyRange(image8bNorm_, threshold, max_threshold)

    if(verbose_):

        mySF.showRangeHist(imageTopTH,hist_values_norm,[threshold,256])


    return imageTopTH

### Apply mast to cut off region
def applyMaskToImage(image,mask):
    copyImage = image.copy()
    if image.shape == mask.shape:
        for i  in range(mask.shape[0]):
            for j  in range(mask.shape[1]):
                if(not mask[i][j]):
                    copyImage[i][j] = 0
                    
        return copyImage
                
        # return cv.bitwise_and(image, mask)  # must then convert images to the same format (uint8 or 16)
    else: 
        print("Error: matrixes must be the same size.")
        return -1




def Gradients(imageGrey,KSIZE):
    
  copyImg = imageGrey.copy()  
  copyImg = cv.GaussianBlur(copyImg, (KSIZE,KSIZE), cv.BORDER_DEFAULT )
  # copyImg = cv.GaussianBlur(copyImg, (KSIZE,KSIZE), cv.BORDER_DEFAULT )

#   sobelx64 = cv.Scharr(imageGrey,cv.CV_16U,1,0)
#   sobely64 = cv.Scharr(imageGrey,cv.CV_16U,0,1)
  
  sobelx64 = cv.Sobel(imageGrey,cv.CV_16U,1,0,ksize=KSIZE)
  sobely64 = cv.Sobel(imageGrey,cv.CV_16U,0,1,ksize=KSIZE)
  combined = cv.addWeighted(sobelx64, 0.5, sobely64, 0.5, 0)
  return combined

def BinGrads(gradsImage, thPercent):
  copyImage = gradsImage.copy()
  max_val = gradsImage.max()
  # copyImage = np.zeros([gradsImage[0],gradsImage[1],cv.CV_16U])
  thvalue = thPercent*(65535)/100
#   thvalue = thPercent*(max_val)/100
  for i  in range(gradsImage.shape[0]):
      for j  in range(gradsImage.shape[1]):
          if( abs(gradsImage[i][j]) > thvalue):
              copyImage[i][j] = 65535
          else :
            copyImage[i][j] = 0  
             
  return copyImage

def substractMask(image, mask,value):
    maskCopy = mask.copy()
    maskCopy = mySF.erode(maskCopy, value)
    maskCopy = applyMaskToImage(image,maskCopy)
    return maskCopy

# def keepConnectedComponents(img, min_size, connectivity=8):
    
#   found = True
#   passed_labels = []
#     # Find all connected ("labels")
#   imgCopy = img.copy()
#     # num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
#     #     img, connectivity=connectivity)
#   num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(imgCopy,connectivity)
    
#     # check size of all connected components (area in pixels)
#   for i in range(num_labels):
#         label_size = stats[i, cv.CC_STAT_AREA]
        
#         # remove connected components smaller than min_size
#         if label_size < min_size:
#             imgCopy[labels == i] = 0
#             num_labels-=1
#         else:
#           if(i != 0):    # dont care of the bg
#             passed_labels.append(i)
    
    

#     # 1 is the minimum value because the background counts
#   if(num_labels <= 1):
#       found = False


def keepConnectedComponents(img, min_size, connectivity=8):
    
  found = False
  
    # Find all connected ("labels")
  imgCopy = img.copy()
    # num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
    #     img, connectivity=connectivity)
  num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(imgCopy,connectivity)
    
    # check size of all connected components (area in pixels)
  for i in range(num_labels):
        label_size = stats[i, cv.CC_STAT_AREA]
        
        # remove connected components smaller than min_size
        if label_size < min_size:
            imgCopy[labels == i] = 0
            num_labels-=1
    
    # 1 is the minimum value because the background counts
  if(num_labels > 1):
      found = True

          
  return imgCopy ,found


# if the original image in that area reaches 
# check if 
          
  return imgCopy, found

# measure depth per area
def validateDepth(connected,image):
  max_grey_value = image.max()
  acc_depth = np.float64(0)
  nof_connected = 0
  min_detected = -1
  imgFlat = image.flatten()
  imgConnect = connected.flatten()
  # find the deepest depth
  # for i in imgCopy[connected > 0]:
  for i in range(len(imgFlat)):
     if imgConnect[i] > 0:
      if(min_detected == -1 or (min_detected>imgFlat[i] and imgFlat[i]>0)):
        min_detected = imgFlat[i]
      nof_connected +=1
      acc_depth += imgFlat[i]
  
  return acc_depth/nof_connected,True


def DetectScratchesLevel(image, mask,th_lvl,min_size,verbose = False):
  bin_percent_th = th_lvl
  grad = Gradients(image,9)
  applyMask = substractMask(grad, mask, 10)
  bin = BinGrads(applyMask, bin_percent_th)
  close = mySF.close(bin,7)
  convert = Convert8b(close)
  connected, check = keepConnectedComponents(convert, min_size, connectivity=8)
  if(check):
    meand, check = validateDepth(connected,image)
  
  if verbose:
   
    fig = plt.figure(figsize=(9, 9))  
    # ax.set_xlabel("Check Cracks")
    fig.add_subplot(2, 3, 1)
    # showing image
    plt.imshow(image,cmap='gray')
    plt.axis('off')
    plt.title("image")
    
    fig.add_subplot(2, 3, 2)
    # showing image
    plt.imshow(grad,cmap='gray')
    plt.axis('off')
    plt.title("Gradients")
    
    fig.add_subplot(2, 3, 3)
    # showing image
    plt.imshow(applyMask,cmap='gray')
    plt.axis('off')
    plt.title("maskBorder Substraction")
    
    fig.add_subplot(2, 3, 4)
    # showing image
    plt.imshow(bin,cmap='gray')
    plt.axis('off')
    plt.title("Binarized by:" +str(bin_percent_th)+ "%")
    
    fig.add_subplot(2, 3, 5)
    # showing image
    plt.imshow(close,cmap='gray')
    plt.axis('off')
    plt.title("closed")
    
    
    fig.add_subplot(2, 3, 6)
    # showing image
    plt.imshow(connected,cmap='gray')
    plt.axis('off')
    plt.title("connected")
      

    plt.show()
  
  return connected, check

def DetectCracksLevel(image, mask, th_lvl,min_size,verbose = False):
  bin_percent_th = th_lvl
  
  grad = Gradients(image,7)
  applyMask = substractMask(grad, mask, 10)
  bin = BinGrads(applyMask, bin_percent_th)
  close = mySF.close(bin,7)
  convert = Convert8b(close)
  connected, check = keepConnectedComponents(convert, min_size, connectivity=8)
  meand, check2 = validateDepth(connected,image)
  
  if verbose:
        

    fig = plt.figure(figsize=(9, 9))  
    # ax.set_xlabel("Check Cracks")
    fig.add_subplot(2, 3, 1)
    # showing image
    plt.imshow(image,cmap='gray')
    plt.axis('off')
    plt.title("image")
    
    fig.add_subplot(2, 3, 2)
    # showing image
    plt.imshow(grad,cmap='gray')
    plt.axis('off')
    plt.title("Gradients")
    
    fig.add_subplot(2, 3, 3)
    # showing image
    plt.imshow(convert,cmap='gray')
    plt.axis('off')
    plt.title("maskBorder Substraction")
    
    fig.add_subplot(2, 3, 4)
    # showing image
    plt.imshow(bin,cmap='gray')
    plt.axis('off')
    plt.title("Binarized by:" +str(bin_percent_th)+ "%")
    
    fig.add_subplot(2, 3, 5)
    # showing image
    plt.imshow(close,cmap='gray')
    plt.axis('off')
    plt.title("closed")
    
    
    fig.add_subplot(2, 3, 6)
    # showing image
    plt.imshow(connected,cmap='gray')
    plt.axis('off')
    plt.title("connected")
      

    plt.show()
          
  return connected, check



# def getUpperLevelGreyValue(ima):
# # will return 3 values indicating the depth of [bg, middleboards,topboards]
    
#     arr_ = []
#     # cluster model
#     copyIma = ima.copy()
#     scale_percent = 30

#     # image8bNorm = Convert8bAndNormalize ( ima ) 
#     width = int(copyIma.shape[1] * scale_percent / 100)
#     height = int(copyIma.shape[0] * scale_percent / 100)
#     dim = (width, height)
#     # resize image to a 30% the original slcale
#     resized = cv.resize(copyIma, dim)
    
#     # validate not null - skipped
#     arr_flatten = resized.flatten()
#     kmeans = KMeans(n_clusters=3, init='k-means++', random_state=0, max_iter=1)
#     kmeans.fit(arr_flatten.reshape(-1,1)) 
#     arr_.append(sorted(np.array(kmeans.cluster_centers_)))
            
#     # convert from list to array    
#     arr_ = np.array(arr_)
    
#     nof_images = np.array(arr_).shape[0]
#     nof_centers = np.array(arr_).shape[1]
    
#     avr_centers = np.sum(arr_,axis=0)/nof_images

#     # round and return
#     return np.rint(sorted(avr_centers))






def getUpperLevelGreyValue(ima, mask):
    # will return 3 values indicating the depth of [bg, middleboards,topboards]
    
    
    arr_ = []
    # cluster model
    copyIma = ima.copy()
    nof_samples = 0
    acc_depth = np.float64(0)
    
    masked = applyMaskToImage(copyIma,mask)
    imgFlat = masked.flatten()
    
    maxa = imgFlat.max()
    mina = imgFlat.min()
    for i in range(len(imgFlat)):
      if imgFlat[i] > 0:
        if (imgFlat[i] < 0):
              print("shouldnt be")
        # if(min_detected == -1 or (min_detected>imgFlat[i] and imgFlat[i]>0)):
        #   min_detected = imgFlat[i]
        nof_samples +=1
        acc_depth += imgFlat[i]
    
    # validate not null - skipped
    


    # round and return
    return acc_depth/nof_samples











# path to the images
TOP_IMAGES_PATH = "./camera_top"
GREY_CALIB_IMAGES_PATH = "./greyLevelCalib"


BOTTOM_IMAGES_PATH = "./camera_bottom/"


scratch_label = 100
crack_label = 101

# areas of interest for each image
area_B58_scratch = [BOTTOM_IMAGES_PATH+"B58.tif", [[1820,927]]]  #  [1780,927]
area_D59_scratch = [BOTTOM_IMAGES_PATH+"D59.tif",[[1078,440], [1037,1088], [1000,1622]] ]

area_B78_crack  = [BOTTOM_IMAGES_PATH+"B78.tif",[[800,980 ]]]  # [800,920    ]]
area_D82_crack  = [BOTTOM_IMAGES_PATH+"D82.tif",[[274,1300 ]]]
area_D83_crack  = [BOTTOM_IMAGES_PATH+"D83.tif",[[318,697 ]]]
area_E24_crack  = [BOTTOM_IMAGES_PATH+"E24.tif",[[665,1614 ]]]
area_E56_crack  = [BOTTOM_IMAGES_PATH+"E56.tif",[[540,1000 ]]]
area_E70_crack  = [BOTTOM_IMAGES_PATH+"E70.tif",[[1730,1440]]]

training_scratch_list = [area_D59_scratch]
training_crack_list = [area_B78_crack,area_D82_crack,area_E24_crack]
# image + labels,... there is an issue here, not knowing which is which in case of crack and scratches in the same board
multiAreaList = [area_D59_scratch,area_B78_crack]
test_list = [area_D83_crack,area_E56_crack,area_E70_crack,area_B58_scratch]
all_list = [area_B58_scratch,area_D59_scratch,area_B78_crack,area_D82_crack,area_D83_crack,area_E24_crack,area_E56_crack,area_E70_crack]
scratch_list = [area_B58_scratch,area_D59_scratch]
crack_list = [area_B78_crack,area_D82_crack,area_D83_crack,area_E24_crack,area_E56_crack,area_E70_crack]

# test_list = [[area_D83_crack,[crack_label]],[area_E56_crack,[crack_label]],[area_E70_crack,[crack_label]],[area_B58_scratch,[scratch_label]]]


###########




# sample scratch
scract_sample_filepath = area_B58_scratch[0]
scract_sample_location = area_B58_scratch[1][0]
scract_image16 = cv.imread(scract_sample_filepath, -1)

crack_sample_filepath = area_B78_crack[0]
crack_sample_location = area_B78_crack[1][0]
crack_image16 = cv.imread(crack_sample_filepath, -1)



scract_hist_values = cv.calcHist([scract_image16.flatten()],channels=[0],mask=None,histSize=[65535],ranges=[0,65535])
scract_image8bNorm = Convert8bAndNormalize ( scract_image16 )
scract_hist_values_norm = cv.calcHist([scract_image8bNorm.flatten()],channels=[0],mask=None,histSize=[256],ranges=[0,255])

crack_hist_values = cv.calcHist([crack_image16.flatten()],channels=[0],mask=None,histSize=[65535],ranges=[0,65535])
crack_image8bNorm = Convert8bAndNormalize ( crack_image16 )
crack_hist_values_norm = cv.calcHist([crack_image8bNorm.flatten()],channels=[0],mask=None,histSize=[256],ranges=[0,255])


#####


scract_sample_filepath = area_B58_scratch[0]
crack_sample_filepath = area_B78_crack[0]

# GET CLUSTER LEVELS
calib_images =[scract_image16]
clusters_centers_scratch = getDepthLayersCenters(calib_images)
calib_images =[crack_image16]
clusters_centers_crack = getDepthLayersCenters(calib_images)

# This is a mask, i expect that at least scratches are included, otherwise we might have to use close/morphology operations
imageTopTH_scratch = getHigherLayer(scract_image8bNorm, clusters_centers_scratch,False)
imageTopTH_crack = getHigherLayer(crack_image8bNorm, clusters_centers_crack,False)


# SCRATCLVL = getUpperLevelGreyValue(scract_image16, imageTopTH_scratch)
# crackLVL = getUpperLevelGreyValue(crack_image16, imageTopTH_crack)


# close holes in the mask
imageTopTH_scratch = mySF.close(imageTopTH_scratch,3)
imageTopTH_crack = mySF.close(imageTopTH_crack,3)







###########


distance = 60

marked_scratch = scract_image16.copy() 
marked_crack = crack_image16.copy() 
marked_scratch = cv.rectangle(marked_scratch, [scract_sample_location[0]-distance,scract_sample_location[1]-distance], [scract_sample_location[0]+distance,scract_sample_location[1]+distance], [150], 10)
marked_crack = cv.rectangle(marked_crack, [crack_sample_location[0]-distance,crack_sample_location[1]-distance], [crack_sample_location[0]+distance,crack_sample_location[1]+distance], [150], 10)

cutoff_scratch_mask = mySF.getImageCutoffAroundPoint(imageTopTH_scratch,scract_sample_location,distance)
cutoff_crack_mask = mySF.getImageCutoffAroundPoint(imageTopTH_crack,crack_sample_location,distance)

cutoff_scratch = mySF.getImageCutoffAroundPoint(scract_image16,scract_sample_location,distance)
cutoff_crack = mySF.getImageCutoffAroundPoint(crack_image16,crack_sample_location,distance)

# mySF.showQuickImage(cutoff_scratch_mask, "mask")
# mySF.showQuickImage(cutoff_crack_mask, "mask")

MaskApplied_scratch = applyMaskToImage(cutoff_scratch,cutoff_scratch_mask)
MaskApplied_crack = applyMaskToImage(cutoff_crack,cutoff_crack_mask)

# mySF.show2ImagesSideBySide(marked_scratch,cutoff_scratch,"Scratch analysis area")
# mySF.show2ImagesSideBySide(marked_crack,cutoff_crack,"Crack analysis area")



# board_grey_level = getUpperLevelGreyValue(crack_image16)

# print("Testing >> finding scratches in scratch image..." )
# FINAL_1, found = DetectScratchesLevel(MaskApplied_scratch,cutoff_scratch_mask,15,75,True)
# print("Detected :"+ str(found) )
# print("Testing >> finding scratches in crack image..." )
# FINAL_2, found = DetectScratchesLevel(MaskApplied_crack,cutoff_crack_mask,15,75,True)
# print("Detected :"+ str(found) )
print("Testing >> finding cracks in crack image..." )
# FINAL_1, found = DetectCracksLevel(MaskApplied_scratch,cutoff_scratch_mask,30,75,True)
# print("Detected :"+ str(found) )
print("Testing >> finding cracks in crack image..." )
FINAL_2, found = DetectCracksLevel(MaskApplied_crack,cutoff_crack_mask,30,25,True)
print("Detected :"+ str(found) )
