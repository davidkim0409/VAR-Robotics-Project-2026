import cv2
import numpy as np

from .features import *

def get_priority_map(image_name):
    """
    Gets the priority map using 6 feature maps.
    Sets different weights on the feature maps according to their importance.
    """

    image_path = f"../data/input_images/{image_name}"

    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get feature maps
    canny_raw  = get_canny(gray)

    f_canny    = canny_raw.astype(float) / 255.0
    f_sobel    = get_sobel(gray).astype(float) / 255.0
    f_saliency = get_saliency(image).astype(float) / 255.0
    f_laplace  = get_laplacian(gray).astype(float) / 255.0
    f_corners  = get_corners(gray).astype(float) / 255.0
    f_dist     = get_distance_fill(canny_raw).astype(float) / 255.0

    # Weight distribution: [canny, sobel, laplace, corners] = [0.4, 0.2, 0.2, 0.2]
    priority = (
        (0.50 * f_canny) +
        (0.20 * f_sobel) + 
        (0.20 * f_laplace) +
        (0.10 * f_corners)
    )

    # If saliency is below 0.1, it's most likely a useless feature (shadows)
    f_saliency[f_saliency < 0.1] = 0.0 

    # Saliency masking
    priority *= f_saliency

    # Distance fitting
    priority += 0.05 * f_dist

    return np.clip(priority, 0, 1)
