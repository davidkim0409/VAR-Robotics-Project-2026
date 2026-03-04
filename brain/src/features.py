import cv2
import numpy as np

def normalize_map(f_map):
    """Utility to ensure every feature map is consistently 0-255 uint8"""
    if f_map.max() > 0:
        f_map = cv2.normalize(f_map, None, 0, 255, cv2.NORM_MINMAX)

    return f_map.astype("uint8")

# --- Features ---

def get_canny(image):
    """Detects Canny Edge"""
    edges = cv2.Canny(image, 75, 150)
    
    return edges

def get_sobel(image):
    """Detects Sobel Edge"""
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges
    
    # Compute gradient magnitude
    gradient_magnitude = cv2.magnitude(sobelx, sobely)
    
    return normalize_map(gradient_magnitude)

def get_saliency(image):
    """Gets the Saliency map"""
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(image)
    saliencyMap = (saliencyMap * 255)

    return normalize_map(saliencyMap)

def get_laplacian(image):
    """Detects high-frequency textures"""
    lap = cv2.Laplacian(image, cv2.CV_64F)
    lap = np.absolute(lap)

    return normalize_map(lap)

def get_corners(image):
    """Detects corners"""
    t_image = np.float32(image) # CornerHarris works best on float32
    dst = cv2.cornerHarris(t_image, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)

    return normalize_map(dst)

def get_distance_fill(edges):
    """Calculates furthest distance from edges"""
    dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)

    return normalize_map(dist)