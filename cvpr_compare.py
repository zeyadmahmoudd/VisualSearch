import numpy as np
import cv2
from scipy.spatial.distance import hamming, jensenshannon, cdist, mahalanobis
from scipy.linalg import inv

def cvpr_compare(F1, F2):
    # This function should compare F1 to F2 - i.e. compute the distance
    # between the two descriptors
    # For now it just returns a random number
    dst = np.sqrt(np.sum(np.square(F1-F2)))
    return dst

def cosine_similarity_compare(desc1, desc2):
    """Best for SIFT, SURF, HOG, GIST"""
    desc1 = desc1.flatten()
    desc2 = desc2.flatten()
    dot_product = np.dot(desc1, desc2)
    norm1 = np.linalg.norm(desc1)
    norm2 = np.linalg.norm(desc2)
    return dot_product / (norm1 * norm2) 

def hamming_similarity_compare(desc1, desc2):
    """Best for BRIEF, BRISK, ORB"""
    # Convert Hamming distance to similarity
    hamming_dist = cv2.countNonZero(cv2.bitwise_xor(desc1, desc2))
    return 1 - (hamming_dist / len(desc1))

def intersection_similarity_compare(desc1, desc2):
    """Best for LBP and Color Histograms"""
    desc1 = desc1.flatten()
    desc2 = desc2.flatten()
    # Normalize if not already normalized
    desc1 = desc1 / np.sum(desc1)
    desc2 = desc2 / np.sum(desc2)
    return np.sum(np.minimum(desc1, desc2))

def bhattacharyya_similarity_compare(desc1, desc2):
    """Best for LBP and Color Histograms"""
    desc1 = desc1.flatten()
    desc2 = desc2.flatten()
    # Normalize if not already normalized
    desc1 = desc1 / np.sum(desc1)
    desc2 = desc2 / np.sum(desc2)
    return np.sum(np.sqrt(desc1 * desc2))

def cross_correlation_compare(desc1, desc2):
    """Alternative for SIFT, SURF"""
    desc1 = desc1.flatten()
    desc2 = desc2.flatten()
    # Normalize to zero mean and unit variance
    desc1 = (desc1 - np.mean(desc1)) / np.std(desc1)
    desc2 = (desc2 - np.mean(desc2)) / np.std(desc2)
    return np.correlate(desc1, desc2)[0] / len(desc1)

def jensenshannon_compare(desc1, desc2):

    desc1 = desc1.flatten()
    desc2 = desc2.flatten()
    
    # Ensure non-negative values
    desc1 = np.abs(desc1)
    desc2 = np.abs(desc2)
    
    # Add small epsilon and normalize to create proper probability distributions
    eps = 1e-10
    desc1 = desc1 + eps
    desc2 = desc2 + eps
    
    # Normalize to sum to 1
    desc1 = desc1 / np.sum(desc1)
    desc2 = desc2 / np.sum(desc2)
    
    # Compute Jensen-Shannon distance
    return jensenshannon(desc1, desc2)

def chi_square_compare(desc1, desc2):
    desc1 = desc1.flatten()
    desc2 = desc2.flatten()
    
    # Ensure positive values and add small epsilon to prevent division by zero
    eps = 1e-10
    desc1 = np.abs(desc1) + eps
    desc2 = np.abs(desc2) + eps
    
    # Compute chi-square distance
    numerator = (desc1 - desc2) ** 2
    denominator = desc1 + desc2
    
    return 0.5 * np.sum(numerator / denominator)

def hamming_compare(desc1, desc2):
    return hamming(desc1, desc2)

def mahalanobis_compare(desc1, desc2, inv_conv):
    """
    Compute Mahalanobis distance between two descriptors
    
    Parameters:
    -----------
    desc1, desc2 : numpy.ndarray
        Descriptors to compare
    all_descriptors : numpy.ndarray
        Matrix of all descriptors from dataset
    
    Returns:
    --------
    float : Mahalanobis distance
    """
    # Ensure descriptors are 1D
    desc1 = desc1.flatten()
    desc2 = desc2.flatten()

    return mahalanobis(desc1, desc2, inv_conv)


def flann_compare(query_desc, database_desc):
    """
    Simplified FLANN matching with robust distance calculation
    """
    try:
        # Convert to float32
        query_desc = query_desc.astype(np.float32)
        database_desc = database_desc.astype(np.float32)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        # Initialize FLANN matcher
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Find matches
        matches = flann.knnMatch(query_desc, database_desc, k=2)
        
        # Apply ratio test
        good_matches = []
        for match_tuple in matches:
            if len(match_tuple) == 2:
                m, n = match_tuple
                if m.distance < 0.6 * n.distance:
                    good_matches.append(m.distance)
        
        # Simple scoring
        if len(good_matches) >= 10:
            return np.mean(good_matches)
        return float('inf')
        
    except Exception as e:
        print(f"Error in FLANN matching: {str(e)}")
        return float('inf')

