
import numpy as np
import cv2

def extractRandom(img):
    # Generate a random row vector with 30 elements
    F = np.random.rand(1, 30)
    # Returns a row [rand rand .... rand] representing an image descriptor
    # computed from image 'img'
    # Note img is expected to be a normalized RGB image (colors range [0,1] not [0,255])
   
    return F


def globalColorDescriptor(img):
     
    red = img[:, :, 0] 
    red = red.reshape(1, -1)
    average_red = np.mean(red) 

    green = img[:, :, 1] 
    green = green.reshape(1, -1)
    average_green = np.mean(green) 

    blue = img[:, :, 2] 
    blue = blue.reshape(1, -1)
    average_blue = np.mean(blue) 

    F=np.array([average_red,  average_green,  average_blue])

    return F

def globalColorHistogram(img, bins=(8, 8, 8)):

    img = (img * 255).astype(np.uint8)

    hist = cv2.calcHist([img], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, None).flatten()
    
    return hist