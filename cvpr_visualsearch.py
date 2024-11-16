import cv2
from random import randint
from cvpr_compare import cvpr_compare, jensenshannon_compare, cosine_similarity_compare, hamming_similarity_compare, intersection_similarity_compare, cross_correlation_compare, bhattacharyya_similarity_compare, chi_square_compare, hamming_compare, flann_compare, mahalanobis_compare
from utils import visualization, inv_conv
import ipdb
import matplotlib.pyplot as plt
from dataset import load_dataset, labelling 
from evaluation import precision_and_recall, plotPRcurve, confusionMatrix


DESCRIPTOR_FOLDER = 'descriptors'
DESCRIPTOR_SUBFOLDER = 'sift'
#DESCRIPTOR_SUBFOLDER = 'hog\pca_reduced'
IMAGE_FOLDER = 'MSRC_ObjCategImageDatabase_v2'

ALLFEAT, ALLFILES = load_dataset(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER, IMAGE_FOLDER)
labels = labelling(ALLFILES)

# Pick a random image as the query
NIMG = ALLFEAT.shape[0]
queryimg_index = randint(0, NIMG - 1)

# Compute the distance between the query and all other descriptors
dst = []
queryimg = ALLFEAT[queryimg_index]

descriptors_inv_conv = inv_conv(ALLFEAT)

for i in range(NIMG):
    candidate = ALLFEAT[i]
    distance = cosine_similarity_compare(queryimg, candidate)
    #distance = mahalanobis_compare(queryimg, candidate, descriptors_inv_conv)
    dst.append((distance, i))

# Sort the distances
dst.sort(key=lambda x: x[0], reverse=True)
#dst.sort(key=lambda x: x[0])

SHOW = 15

precision, recall, labels_by_rank = precision_and_recall(ALLFILES, queryimg_index, dst, labels)

visualization(ALLFILES, queryimg_index, dst, SHOW, precision, recall)
plotPRcurve(precision, recall, SHOW)
confusionMatrix(ALLFILES, queryimg_index, dst, SHOW, labels_by_rank)

cv2.destroyAllWindows()

