import os
import numpy as np
import scipy.io as sio
import cv2
from random import randint
from cvpr_compare import cvpr_compare
from utils import visualization
import ipdb
import matplotlib.pyplot as plt

DESCRIPTOR_FOLDER = 'descriptors'
DESCRIPTOR_SUBFOLDER = 'globalRGBhisto'
IMAGE_FOLDER = 'MSRC_ObjCategImageDatabase_v2'

# Load all descriptors
ALLFEAT = []
ALLFILES = []
for filename in os.listdir(os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER)):
    if filename.endswith('.mat'):
        img_path = os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER, filename)
        img_actual_path = os.path.join(IMAGE_FOLDER,'Images',filename).replace(".mat",".bmp")
        # ipdb.set_trace()
        img_data = sio.loadmat(img_path)
        ALLFILES.append(img_actual_path)
        ALLFEAT.append(img_data['F'][0])  # Assuming F is a 1D array

# Convert ALLFEAT to a numpy array
ALLFEAT = np.array(ALLFEAT)

# Pick a random image as the query
NIMG = ALLFEAT.shape[0]
queryimg_index = randint(0, NIMG - 1)

# Compute the distance between the query and all other descriptors
dst = []
queryimg = ALLFEAT[queryimg_index]
for i in range(NIMG):
    candidate = ALLFEAT[i]
    distance = cvpr_compare(queryimg, candidate)
    dst.append((distance, i))

# Sort the distances
dst.sort(key=lambda x: x[0])

SHOW = 15
visualization(ALLFILES, queryimg_index, dst, SHOW)

cv2.destroyAllWindows()

