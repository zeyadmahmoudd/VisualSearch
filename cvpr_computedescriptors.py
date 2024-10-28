import os
import numpy as np
import cv2
import scipy.io as sio
from extractDescriptor import extractRandom, globalColorDescriptor, globalColorHistogram, process_grid_cells

DATASET_FOLDER = 'MSRC_ObjCategImageDatabase_v2'
OUT_FOLDER = 'descriptors'
OUT_SUBFOLDER = 'globalRGBhisto'

# Ensure the output directory exists
os.makedirs(os.path.join(OUT_FOLDER, OUT_SUBFOLDER), exist_ok=True)

# Iterate through all BMP files in the dataset folder
for filename in os.listdir(os.path.join(DATASET_FOLDER, 'Images')):
    if filename.endswith(".bmp"):
        print(f"Processing file {filename}")
        img_path = os.path.join(DATASET_FOLDER, 'Images', filename)
        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0  # Normalize the image
        fout = os.path.join(OUT_FOLDER, OUT_SUBFOLDER, filename.replace('.bmp', '.mat'))

        # Call extractRandom (or another feature extraction function) to get the descriptor
        #F = process_grid_cells(img, grid_size = 3, bins=(8, 8, 8), RGB = True, HSV = True, LAB = True, moments = True)
        F = globalColorHistogram(img, "RGB", bins=(8, 8, 8))

        # Save the descriptor to a .mat file
        sio.savemat(fout, {'F': F})

