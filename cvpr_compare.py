
import numpy as np

def cvpr_compare(F1, F2):
    # This function should compare F1 to F2 - i.e. compute the distance
    # between the two descriptors
    # For now it just returns a random number
    dst = np.sqrt(np.sum(np.square(F1-F2)))
    return dst


