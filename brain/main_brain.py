import matplotlib.pyplot as plt
import numpy as np

from src.feature_extractor import get_priority_map
from src.point_sampler import sample_drone_points

def visulaize_sampling(priority_map, sampled_points):    
    """
    Visualizes the sampled points from the priority map.
    """

    plt.figure("Drones", figsize=(8, 8))
    black_canvas = np.zeros((priority_map.shape[0], priority_map.shape[1], 3))
    plt.imshow(black_canvas)

    # Extract coordinates
    x_coords = [p[0] for p in sampled_points]
    y_coords = [p[1] for p in sampled_points]
    
    # Plot drones
    plt.scatter(x_coords, y_coords, c='cyan', s=5, alpha=0.8, edgecolors='none')
    
    plt.title(f"Swarm Distribution: {len(sampled_points)} Drones")
    plt.axis('off')
    plt.show()

def main():
    path = '../data/input_images/son.png'

    num_drones = 1000
    min_dist = 5

    priority_m = get_priority_map(path)
    points = sample_drone_points(priority_m, num_drones, min_dist)

    visulaize_sampling(priority_m, points)

if __name__ == "__main__":
    main()