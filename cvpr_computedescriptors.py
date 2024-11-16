import os
import numpy as np
import cv2
import scipy.io as sio
from extractDescriptor import extractRandom, lbp, BRIEF, FREAK, ORB, globalColorDescriptor, globalColorHistogram, glcm, HOG, process_grid_cells, SURF, SIFT, GIST

DATASET_FOLDER = 'MSRC_ObjCategImageDatabase_v2'
OUT_FOLDER = 'descriptors'
OUT_SUBFOLDER = 'lbp_P16_R2'

# Ensure the output directory exists
os.makedirs(os.path.join(OUT_FOLDER, OUT_SUBFOLDER), exist_ok=True)

# Iterate through all BMP files in the dataset folder
for filename in os.listdir(os.path.join(DATASET_FOLDER, 'Images')):
    if filename.endswith(".bmp"):
        print(f"Processing file {filename}")
        img_path = os.path.join(DATASET_FOLDER, 'Images', filename)
        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0  # Normalize the image
        img = cv2.resize(img, (224, 224))
        fout = os.path.join(OUT_FOLDER, OUT_SUBFOLDER, filename.replace('.bmp', '.mat'))
        
        # Call extractRandom (or another feature extraction function) to get the descriptor
        #F = process_grid_cells(img, grid_size = 3, bins=(8, 8, 8), RGB = True, HSV = True, LAB = True, moments = True)
        #F = globalColorHistogram(img, "RGB", bins=(8, 8, 8))
        # F1 = HOG(img)
        # F2 = global ColorHistogram(img)
        #F = SIFT(img, n_keypoints=500, contrastThreshold = 0.02, edgeThreshold = 10, sigma = 1.6)
        F = lbp(img)
        # Save the descriptor to a .mat file
        sio.savemat(fout, {'F': F})
    

