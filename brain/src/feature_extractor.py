import cv2
import numpy as np

from src.features import *

def get_priority_map(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get feature maps
    f_canny    = get_canny(gray).astype(float) / 255.0
    f_sobel    = get_sobel(gray).astype(float) / 255.0
    f_saliency = get_saliency(image).astype(float) / 255.0
    f_laplace  = get_laplacian(gray).astype(float) / 255.0
    f_corners  = get_corners(gray).astype(float) / 255.0
    f_dist     = get_distance_fill(f_canny).astype(float) / 255.0

    # Weight distribution: [canny, sobel, laplace, corners] = [0.4, 0.2, 0.2, 0.2]
    priority = (
        (0.40 * f_canny) +
        (0.20 * f_sobel) + 
        (0.20 * f_laplace) +
        (0.20 * f_corners)
    )

    # Saliency masking
    priority *= f_saliency

    # Distance fitting
    priority += 0.05 * f_dist

    return np.clip(priority, 0, 1)