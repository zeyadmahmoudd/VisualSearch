
import numpy as np
import cv2
from utils import create_grid
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog


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

def globalColorHistogram(img, color_space = 'RGB', bins=(8, 8, 8)):

    img = (img* 255).astype(np.uint8)

    if color_space == 'RGB':
        hist = cv2.calcHist([img], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])

    elif color_space == 'HSV':
        hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv_image], [0,1,2], None, bins, 
                        [0,180,0,256,0,256])
        
    elif color_space == 'LAB':
        lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        hist = cv2.calcHist([lab_img], [0,1,2], None, bins, 
                           [0,256,0,256,0,256])
        
    hist = cv2.normalize(hist, None, norm_type=cv2.NORM_L1).flatten()
    return hist

def color_moments(img):

    r, g, b = cv2.split(img)
    moments = []

    for channel in (r, g, b):
        mean = np.mean(channel)
        std = np.std(channel)
        skewness = np .mean(((channel - mean)/std)**3) if std!=0 else 0

    moments.extend([mean, std, skewness])

    mean = np.mean(moments)
    std = np.std(moments)
    
    return (moments - mean) / (std + 1e-7)

def lbp(img):
    
    img = (img * 255).astype(np.uint8)

    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    lbp = local_binary_pattern(gray_image, n_points=8, radius=1, method='uniform')

    n_bins = 8 + 2  # For uniform LBP with 8 points
    histogram, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    histogram = histogram / (np.sum(histogram) + 1e-7)
    

    return histogram

def glcm(img, distances = [50], angles = [np.pi/2]): #Distance offset to look at and angles is the directions to look at pi/2 means vertical only

    img = (img * 255).astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    glcm = graycomatrix(img, 
                    distances=distances, 
                    angles=angles,
                    levels=256)
    
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

    feature_vector = []

    for prop in properties:
        feature = graycoprops(glcm, prop).ravel() 
        feature_vector.extend(feature)

    feature_vector = np.array(feature_vector)

    return feature_vector

def HOG(img, orientations=9, pixels_per_cell=8, cells_per_block=2):

    img = (img * 255).astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    hog_features = hog(img, 
                    orientations=orientations, 
                    pixels_per_cell=(pixels_per_cell, pixels_per_cell), 
                    cells_per_block=(cells_per_block, cells_per_block),
                    transform_sqrt=True)
    hog_features = hog_features / np.linalg.norm(hog_features)
    
    return hog_features

def process_grid_cells(img, grid_size, bins=(8, 8, 8), RGB = True, HSV = False, LAB = False, moments = False):

    grid_cells = create_grid(img, grid_size)
    all_features = []

    for cell in grid_cells:
        if RGB:
            features = globalColorHistogram(cell, 'RGB', bins)
            all_features.append(features)
        if HSV:
            features = globalColorHistogram(cell, 'HSV', bins)
            all_features.append(features)
        if LAB:
            features = globalColorHistogram(cell, 'LAB', bins)
            all_features.append(features)
        if color_moments:
            features = color_moments(cell)
            all_features.append(features)

    final_features = np.concatenate(all_features)
    return final_features



