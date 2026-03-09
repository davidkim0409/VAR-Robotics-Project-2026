import matplotlib.pyplot as plt
import numpy as np
import json
import os

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

def visulaize_priority_map(priority_map):
    plt.figure("Priority Map", figsize=(8, 8))
    plt.imshow(priority_map, cmap='magma', vmin=0.0, vmax=1.0)
    plt.colorbar(label="Drone Priority (0.0 to 1.0)")
    plt.title("Priority Map")
    plt.axis('off')
    plt.show()

def export_coords(image_name, num_drones, min_dist):
    input_path = f"../data/input_images/{image_name}"
    base_name = os.path.splitext(image_name)[0] # Remove extension
    
    output_path = f"../data/coordinates/{base_name}_coords.json"

    priority_map = get_priority_map(input_path)
    points = sample_drone_points(priority_map, num_drones=num_drones, min_dist=min_dist)

    coord_data = {
        "image_source": image_name,
        "image_dimensions": priority_map.shape, # [height, width]
        "drone_count": len(points),
        "points": [{"x": p[0], "y": p[1]} for p in points]
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coord_data, f, indent=4)

    print(f"---- {len(points)} coordinates saved to {output_path} ----")

if __name__ == "__main__":
    image_name = "minion.png" 
    num_drones = 200 
    min_dist = 2

    # export_coords(image_name, num_drones, min_dist)
    
    p_map = get_priority_map(image_name)
    points = sample_drone_points(p_map, num_drones, min_dist)
    visulaize_sampling(p_map, points)
    visulaize_priority_map(p_map)