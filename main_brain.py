import matplotlib.pyplot as plt
import cv2
from brain.src.feature_extractor import get_priority_map

def visualize_brain_logic(image_path):
    # 1. Get the final map (values 0.0 to 1.0)
    priority_map = get_priority_map(image_path)
    
    # 2. Display it
    plt.figure(figsize=(10, 8))
    
    # Use 'inferno' or 'magma' colormap—it makes high-priority areas 
    # look like glowing "heat" which is easier to debug than grayscale.
    plt.imshow(priority_map, cmap='magma')
    plt.colorbar(label="Drone Priority (0.0 to 1.0)")
    plt.title("The Priority Map: Where the Swarm wants to be")
    plt.axis('off')
    plt.show()

# Run it on your Minion!
visualize_brain_logic("data/input_images/son.png")
