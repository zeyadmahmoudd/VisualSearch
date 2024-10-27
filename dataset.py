import numpy as np
import os
import scipy.io as sio

def load_dataset(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER, IMAGE_FOLDER):

    # Load all descriptors
    ALLFEAT = []
    ALLFILES = []
    for filename in os.listdir(os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER)):
        if filename.endswith('.mat'):
            img_path = os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER, filename)
            img_actual_path = os.path.join(IMAGE_FOLDER,'Images',filename).replace(".mat",".bmp")
            img_label = filename.split("_")[0]
            # ipdb.set_trace()
            img_data = sio.loadmat(img_path)
            ALLFILES.append([img_actual_path, img_label])
            ALLFEAT.append(img_data['F'][0])  # Assuming F is a 1D array

    # Convert ALLFEAT to a numpy array
    ALLFEAT = np.array(ALLFEAT)

    return ALLFEAT, ALLFILES

def labelling(ALLFILES):
    
    labels = {}
    for i in range(len(ALLFILES)):
        #print(ALLFILES[i][1])
        if ALLFILES[i][1] in labels:
            labels[ALLFILES[i][1]] += 1
        else:
            labels[ALLFILES[i][1]] = 1
    return labels