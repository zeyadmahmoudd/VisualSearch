
import numpy as np
import cv2
from utils import create_grid
from skimage.color import rgb2gray
from scipy import ndimage, signal
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

    F_normalized = (F - np.mean(F)) / (np.std(F) + 1e-7)

    return F_normalized

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

    lbp = local_binary_pattern(gray_image, P=16, R=2, method='uniform')

    n_bins = 8*7 + 2  # For uniform LBP with 8 points
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    return hist

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

def HOG(img, orientations=8, pixels_per_cell=16, cells_per_block=2):

    img = (img * 255).astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    hog_features = hog(img, 
                    orientations=orientations, 
                    pixels_per_cell=(pixels_per_cell, pixels_per_cell), 
                    cells_per_block=(cells_per_block, cells_per_block),
                    transform_sqrt=True)
    hog_features = hog_features / np.linalg.norm(hog_features)
    
    return hog_features

def SIFT(img, n_keypoints=500, contrastThreshold=0.04, edgeThreshold=15, sigma=1.6):
    """
    Enhanced SIFT feature extraction optimized for object matching
    """
    # Preprocess image
    if img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img
        
    # Enhance contrast
    img_gray = cv2.equalizeHist(img_gray)
    
    # Initialize SIFT with tuned parameters
    sift = cv2.SIFT_create(
        nfeatures=n_keypoints,
        contrastThreshold=contrastThreshold,  # Slightly lower to detect more features
        edgeThreshold=edgeThreshold,         # Slightly higher to capture more edge features
        sigma=sigma
    )
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)
    
    if descriptors is None or len(descriptors) == 0:
        # Create default descriptor based on image statistics
        descriptors = np.zeros((1, 128))
        mean_val = np.mean(img_gray)
        std_val = np.std(img_gray)
        descriptors[0, :64] = mean_val
        descriptors[0, 64:] = std_val
    
    # Handle descriptor count
    if len(descriptors) < n_keypoints:
        num_to_duplicate = n_keypoints - len(descriptors)
        indices = np.random.choice(len(descriptors), num_to_duplicate)
        additional_descriptors = descriptors[indices]
        descriptors = np.vstack((descriptors, additional_descriptors))
    else:
        # Sort descriptors by response strength and take top n_keypoints
        responses = [kp.response for kp in keypoints]
        sorted_indices = np.argsort(responses)[::-1][:n_keypoints]
        descriptors = descriptors[sorted_indices]

    return descriptors



def BRIEF(img):
    """
    Detect FAST keypoints and compute BRIEF descriptors
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image (can be RGB or grayscale)
    
    Returns:
    --------
    descriptors : numpy.ndarray
        BRIEF descriptors for the detected FAST keypoints
    """
    # Convert to grayscale if needed
    if img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img
    
    # Initialize FAST detector and BRIEF extractor
    fast = cv2.FastFeatureDetector_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    
    # Detect keypoints using FAST
    keypoints = fast.detect(img_gray, None)
    
    # Compute BRIEF descriptors for the FAST keypoints
    # Note: brief.compute returns the same keypoints with their descriptors
    _, descriptors = brief.compute(img_gray, keypoints)
    
    return descriptors

def BRISK(img):
    """
    Compute BRISK descriptors for an image using FAST keypoints
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image (can be RGB or grayscale)
    
    Returns:
    --------
    descriptors : numpy.ndarray
        BRISK descriptors for the detected keypoints
    """
    # Convert to grayscale if needed
    if img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img
    
    # Initialize FAST detector and BRISK descriptor
    fast = cv2.FastFeatureDetector_create()
    brisk = cv2.BRISK_create()
    
    # Detect keypoints using FAST
    keypoints = fast.detect(img_gray, None)
    
    # Compute BRISK descriptors
    _, descriptors = brisk.compute(img_gray, keypoints)
    
    return descriptors

def FREAK(img):
    """
    Compute FREAK descriptors for an image at given keypoints
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image (can be RGB or grayscale)
    keypoints : list
        List of keypoints where to compute the descriptors
        
    Returns:
    --------
    descriptors : numpy.ndarray
        FREAK descriptors for the provided keypoints
    """
    # Convert to grayscale if needed
    if img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img
    

    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(img_gray, None)

    # Initialize FREAK descriptor
    freak = cv2.xfeatures2d.FREAK_create()
    
    # Compute FREAK descriptors at the provided keypoints
    _, descriptors = freak.compute(img_gray, keypoints)
    
    return descriptors


def ORB(img):
    img = (img * 255).astype(np.uint8)
    gray_image = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)
    return descriptors

class SURF_Descriptor:
    def __init__(self, hessian_threshold=100, n_octaves=4, n_octave_layers=3):
        """
        Initialize SURF detector
        Args:
            hessian_threshold: Threshold for the keypoint detector
            n_octaves: Number of pyramid octaves
            n_octave_layers: Number of layers within each octave
        """
        self.hessian_threshold = hessian_threshold
        self.n_octaves = n_octaves
        self.n_octave_layers = n_octave_layers

    def create_integral_image(self, image):
        """Create integral image for fast box filter computation"""
        return np.cumsum(np.cumsum(image.astype(float), axis=0), axis=1)

    def box_filter(self, integral_img, x, y, box_size):
        """
        Compute box filter response at a given point with proper boundary handling
        Args:
            integral_img: Integral image
            x, y: Point coordinates
            box_size: Size of the box filter
        """
        height, width = integral_img.shape
        half_size = box_size // 2
        
        # Ensure coordinates are within image boundaries
        x = int(min(max(x, half_size), width - half_size - 1))
        y = int(min(max(y, half_size), height - half_size - 1))
        
        # Calculate box boundaries
        x1 = max(0, x - half_size)
        x2 = min(width - 1, x + half_size)
        y1 = max(0, y - half_size)
        y2 = min(height - 1, y + half_size)
        
        # Compute box sum using integral image
        value = integral_img[y2, x2]
        
        if x1 > 0 and y1 > 0:
            value += integral_img[y1-1, x1-1]
        if x1 > 0:
            value -= integral_img[y2, x1-1]
        if y1 > 0:
            value -= integral_img[y1-1, x2]
            
        return value

    def compute_hessian_response(self, integral_img, x, y, size):
        """
        Compute Hessian matrix response at a point
        Args:
            integral_img: Integral image
            x, y: Point coordinates
            size: Filter size
        """
        try:
            # Compute Dxx
            dxx_size = size
            dxx = self.box_filter(integral_img, x, y + size//4, dxx_size) + \
                  self.box_filter(integral_img, x, y - size//4, dxx_size) - \
                  2 * self.box_filter(integral_img, x, y, dxx_size)
                  
            # Compute Dyy
            dyy_size = size
            dyy = self.box_filter(integral_img, x + size//4, y, dyy_size) + \
                  self.box_filter(integral_img, x - size//4, y, dyy_size) - \
                  2 * self.box_filter(integral_img, x, y, dyy_size)
                  
            # Compute Dxy
            dxy_size = size//2
            dxy = self.box_filter(integral_img, x + dxy_size, y + dxy_size, dxy_size) + \
                  self.box_filter(integral_img, x - dxy_size, y - dxy_size, dxy_size) - \
                  self.box_filter(integral_img, x + dxy_size, y - dxy_size, dxy_size) - \
                  self.box_filter(integral_img, x - dxy_size, y + dxy_size, dxy_size)
                  
            # Compute determinant
            det = (dxx * dyy) - (0.81 * dxy * dxy)  # 0.81 is SURF's magic number
            
            return det
        except Exception as e:
            print(f"Error in compute_hessian_response: {str(e)}")
            return 0

    def detect_keypoints(self, integral_img):
        """
        Detect keypoints using Hessian matrix with proper boundary handling
        Args:
            integral_img: Integral image
        Returns:
            List of keypoints (x, y, size, response)
        """
        keypoints = []
        height, width = integral_img.shape
        
        # Initial filter size
        initial_size = 9
        
        # For each octave
        for octave in range(self.n_octaves):
            size = initial_size * (2 ** octave)
            step = max(1, size // 2)  # Sampling step
            
            # Adjust bounds to prevent boundary issues
            border = size * 2
            
            # For each point in the image
            for y in range(border, height - border, step):
                for x in range(border, width - border, step):
                    try:
                        # Compute Hessian response
                        response = self.compute_hessian_response(integral_img, x, y, size)
                        
                        # If response is above threshold
                        if response > self.hessian_threshold:
                            # Non-maximum suppression in local neighborhood
                            is_maximum = True
                            for dy in [-step, 0, step]:
                                for dx in [-step, 0, step]:
                                    if dx == 0 and dy == 0:
                                        continue
                                        
                                    neighbor_x = x + dx
                                    neighbor_y = y + dy
                                    
                                    if (border <= neighbor_x < width - border and 
                                        border <= neighbor_y < height - border):
                                        neighbor_response = self.compute_hessian_response(
                                            integral_img, neighbor_x, neighbor_y, size
                                        )
                                        if neighbor_response >= response:
                                            is_maximum = False
                                            break
                                            
                                if not is_maximum:
                                    break
                                    
                            if is_maximum:
                                keypoints.append((x, y, size, response))
                    except Exception as e:
                        print(f"Error processing point ({x}, {y}): {str(e)}")
                        continue
                        
        return keypoints
    
    def compute_orientation(self, integral_img, keypoint):
        """
        Compute dominant orientation for a keypoint with boundary checking
        Args:
            integral_img: Integral image
            keypoint: (x, y, size, response)
        Returns:
            Orientation in radians
        """
        try:
            x, y, size, _ = keypoint
            height, width = integral_img.shape
            radius = min(size * 6, min(width//2, height//2))  # Limit radius to image size
            
            # Compute Haar wavelet responses in x and y
            responses_x = []
            responses_y = []
            
            for dy in range(-radius, radius + 1, size):
                for dx in range(-radius, radius + 1, size):
                    # Skip points outside circular region
                    if dx*dx + dy*dy > radius*radius:
                        continue
                        
                    # Check if sample point is within image boundaries
                    sample_x = x + dx
                    sample_y = y + dy
                    
                    if (size//2 <= sample_x < width - size//2 and 
                        size//2 <= sample_y < height - size//2):
                        # Compute wavelet responses
                        haar_x = self.box_filter(integral_img, sample_x + size//2, sample_y, size) - \
                                self.box_filter(integral_img, sample_x - size//2, sample_y, size)
                                
                        haar_y = self.box_filter(integral_img, sample_x, sample_y + size//2, size) - \
                                self.box_filter(integral_img, sample_x, sample_y - size//2, size)
                                
                        responses_x.append(haar_x)
                        responses_y.append(haar_y)
            
            if not responses_x:  # If no valid responses found
                return 0.0
                
            # Find dominant orientation
            angles = np.arctan2(responses_y, responses_x)
            orientations = np.linspace(-np.pi, np.pi, 36)  # 36 bins
            
            max_sum = 0
            dominant_orientation = 0
            
            for orientation in orientations:
                # Sum responses within 60 degree window
                window = np.pi/3
                in_window = np.abs(angles - orientation) < window
                sum_x = np.sum(np.array(responses_x)[in_window])
                sum_y = np.sum(np.array(responses_y)[in_window])
                
                # Compute magnitude
                magnitude = np.sqrt(sum_x*sum_x + sum_y*sum_y)
                
                if magnitude > max_sum:
                    max_sum = magnitude
                    dominant_orientation = orientation
                    
            return dominant_orientation
            
        except Exception as e:
            print(f"Error in compute_orientation: {str(e)}")
            return 0.0

    def compute_descriptor(self, integral_img, keypoint, orientation):
        """
        Compute SURF descriptor for a keypoint
        Args:
            integral_img: Integral image
            keypoint: (x, y, size, response)
            orientation: Keypoint orientation
        Returns:
            64-dimensional descriptor
        """
        try:
            x, y, size, _ = keypoint
            height, width = integral_img.shape
            
            # Initialize descriptor array
            descriptor = np.zeros(64)
            
            # Rotate grid by keypoint orientation
            cos_t = np.cos(orientation)
            sin_t = np.sin(orientation)
            
            # Descriptor parameters
            grid_spacing = size * 5
            
            # 4x4 grid
            desc_idx = 0
            for iy in range(-2, 2):
                for ix in range(-2, 2):
                    # Get rotated grid coordinates
                    grid_x = (ix * grid_spacing)
                    grid_y = (iy * grid_spacing)
                    
                    rx = grid_x * cos_t - grid_y * sin_t + x
                    ry = grid_x * sin_t + grid_y * cos_t + y
                    
                    # Compute Haar wavelet responses in sub-region
                    dx_sum = 0
                    dy_sum = 0
                    abs_dx_sum = 0
                    abs_dy_sum = 0
                    count = 0
                    
                    # Sample points in sub-region
                    for sy in range(-grid_spacing//2, grid_spacing//2, size):
                        for sx in range(-grid_spacing//2, grid_spacing//2, size):
                            # Get rotated sample point
                            sample_x = int(rx + (sx * cos_t - sy * sin_t))
                            sample_y = int(ry + (sx * sin_t + sy * cos_t))
                            
                            # Check if point is within image boundaries
                            if (size <= sample_x < width - size and 
                                size <= sample_y < height - size):
                                # Compute wavelet responses
                                dx = self.box_filter(integral_img, sample_x + size, sample_y, size) - \
                                     self.box_filter(integral_img, sample_x - size, sample_y, size)
                                dy = self.box_filter(integral_img, sample_x, sample_y + size, size) - \
                                     self.box_filter(integral_img, sample_x, sample_y - size, size)
                                     
                                dx_sum += dx
                                dy_sum += dy
                                abs_dx_sum += abs(dx)
                                abs_dy_sum += abs(dy)
                                count += 1
                    
                    # Add sub-region values to descriptor
                    if count > 0:
                        descriptor[desc_idx:desc_idx+4] = [dx_sum/count, 
                                                         dy_sum/count, 
                                                         abs_dx_sum/count, 
                                                         abs_dy_sum/count]
                    desc_idx += 4
            
            # Normalize descriptor
            norm = np.linalg.norm(descriptor)
            if norm > 0:
                descriptor = descriptor / norm
                
            return descriptor
            
        except Exception as e:
            print(f"Error in compute_descriptor: {str(e)}")
            return np.zeros(64)

    def detect_and_compute(self, image):
        """
        Detect keypoints and compute their descriptors
        Args:
            image: Input image (grayscale)
        Returns:
            keypoints: List of keypoints
            descriptors: Array of descriptors
        """
        try:
            # Create integral image
            integral_img = self.create_integral_image(image)
            
            # Detect keypoints
            keypoints = self.detect_keypoints(integral_img)
            
            if not keypoints:
                print("No keypoints detected")
                return None, np.zeros((1, 64))
                
            # Compute orientations and descriptors
            descriptors = []
            final_keypoints = []
            
            for keypoint in keypoints:
                # Compute orientation
                orientation = self.compute_orientation(integral_img, keypoint)
                
                # Compute descriptor
                descriptor = self.compute_descriptor(integral_img, keypoint, orientation)
                
                if descriptor is not None:
                    descriptors.append(descriptor)
                    final_keypoints.append(keypoint + (orientation,))
                    
            if not descriptors:
                print("No valid descriptors computed")
                return None, np.zeros((1, 64))
                
            return np.array(final_keypoints), np.array(descriptors)
            
        except Exception as e:
            print(f"Error in detect_and_compute: {str(e)}")
            return None, np.zeros((1, 64))

def SURF(img):
    """
    Main SURF function with proper error handling
    Args:
        img: Input image (RGB, normalized to [0,1], size 224x224)
    Returns:
        numpy.ndarray: SURF descriptors
    """
    try:
        # Convert to uint8
        img_uint8 = (img * 255).astype(np.uint8)
        
        # Convert RGB to grayscale
        if len(img_uint8.shape) == 3:
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_uint8
        
        # Create SURF detector
        surf = SURF_Descriptor(hessian_threshold=100)
        
        # Detect and compute
        _, descriptors = surf.detect_and_compute(gray)
        
        if descriptors is None:
            return np.zeros((1, 64))
            
        return descriptors
        
    except Exception as e:
        print(f"Error in SURF computation: {str(e)}")
        return np.zeros((1, 64))
    







def create_gabor_filters(n_orientations=8, n_scales=4):
    """
    Create Gabor filters with different scales and orientations
    """
    filters = []
    for scale in range(n_scales):
        for orientation in range(n_orientations):
            sigma = 3 * 2**scale
            wavelength = sigma * 2
            theta = orientation * np.pi / n_orientations
            
            # Create meshgrid for the filter
            size = int(2.5 * sigma)
            if size % 2 == 0:
                size += 1
            x, y = np.meshgrid(np.arange(-size//2, size//2 + 1),
                             np.arange(-size//2, size//2 + 1))
            
            # Rotation
            x_theta = x * np.cos(theta) + y * np.sin(theta)
            y_theta = -x * np.sin(theta) + y * np.cos(theta)
            
            # Gabor function
            gb = np.exp(-.5 * (x_theta**2 + y_theta**2) / sigma**2) * \
                 np.cos(2 * np.pi * x_theta / wavelength)
            
            filters.append(gb)
    
    return filters

def GIST(img, n_blocks=4, n_orientations=8, n_scales=4):
    """
    Compute GIST descriptor for an input image
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image (RGB, normalized [0,1], size 224x224)
    n_blocks : int
        Number of blocks to divide the image into (n_blocks x n_blocks)
    n_orientations : int
        Number of orientations for Gabor filters
    n_scales : int
        Number of scales for Gabor filters
    
    Returns:
    --------
    numpy.ndarray
        GIST descriptor vector
    """
    # Convert RGB to grayscale using luminance formula
    img = (img * 255).astype(np.uint8)
    gray_image = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
    # Normalize the grayscale image
    gray_image = (gray_image - np.mean(gray_image)) / (np.std(gray_image) + 1e-8)
    
    # Create Gabor filters
    filters = create_gabor_filters(n_orientations, n_scales)
    
    # Apply filters
    filtered_images = []
    for gb in filters:
        filtered = signal.convolve2d(gray_image, gb, mode='same', boundary='wrap')
        filtered_images.append(filtered)
    
    # Divide each filtered image into blocks and compute average
    block_size = (gray_image.shape[0] // n_blocks, gray_image.shape[1] // n_blocks)
    features = []
    
    for filtered in filtered_images:
        for i in range(n_blocks):
            for j in range(n_blocks):
                block = filtered[i*block_size[0]:(i+1)*block_size[0],
                               j*block_size[1]:(j+1)*block_size[1]]
                features.append(np.mean(np.abs(block)))
    
    return np.array(features)




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
        if moments:
            features = color_moments(cell)
            all_features.append(features)

    final_features = np.concatenate(all_features)
    
    return final_features



