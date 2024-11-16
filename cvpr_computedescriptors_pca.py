# pca_reduction.py
import os
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
import joblib


OUT_FOLDER = 'descriptors'
OUT_SUBFOLDER = 'sift'
DESCRIPTOR_FOLDER = os.path.join(OUT_FOLDER, OUT_SUBFOLDER)  

def apply_pca_to_descriptors(descriptor_folder, n_components=100):
    descriptors = []
    filenames = []
    
    print("Loading descriptors...")
    for filename in os.listdir(descriptor_folder):
        if filename.endswith('.mat'):
            mat_file = sio.loadmat(os.path.join(descriptor_folder, filename))
            descriptor = mat_file['F'].flatten()
            descriptors.append(descriptor)
            filenames.append(filename)
    
    descriptors = np.array(descriptors)
    print(f"Loaded {len(descriptors)} descriptors of shape {descriptors.shape}")
    
    print("Applying PCA...")
    pca = PCA(n_components=n_components)
    descriptors_reduced = pca.fit_transform(descriptors)
    
    print(f"Reduced dimensions from {descriptors.shape[1]} to {descriptors_reduced.shape[1]}")
    print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.3f}")
    
    out_folder = os.path.join(descriptor_folder, 'pca_reduced')
    os.makedirs(out_folder, exist_ok=True)
    
    print("Saving reduced descriptors...")
    for i, filename in enumerate(filenames):
        fout = os.path.join(out_folder, filename)
        sio.savemat(fout, {'F': descriptors_reduced[i]})
    
    model_path = os.path.join(out_folder, 'pca_model.pkl')
    joblib.dump(pca, model_path)
    print(f"PCA model saved to {model_path}")
    
    return descriptors_reduced, pca


reduced_features, pca_model = apply_pca_to_descriptors(DESCRIPTOR_FOLDER, n_components=100)