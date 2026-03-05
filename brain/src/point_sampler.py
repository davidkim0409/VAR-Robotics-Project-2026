import numpy as np
import random

def sample_drone_points(priority_map, num_drones=500, min_dist=5):
    """
    Samples pixels from the priority map. 
    Returns a list of (x, y) tuples in pixel coordinates.
    """
    height, width = priority_map.shape
    sampled_points = []
    
    # Use a high attempt limit to ensure we hit the target count 
    # even with strict distance constraints.
    max_attempts = num_drones * 150 
    attempts = 0

    threshold = 0.05 # Threshold to avoid sampling from blank spots
    
    while len(sampled_points) < num_drones and attempts < max_attempts:
        attempts += 1
        
        # Randomly pick a candidate pixel
        ry = random.randint(0, height - 1)
        rx = random.randint(0, width - 1)
        
        priority = priority_map[ry, rx]
        
        # If the sampled point is below threshold, ignore it
        if priority < threshold:
            continue

        # Probability Test
        # The higher the priority, the more likely this point is accepted.
        if random.random() > priority:
            continue
            
        # Dynamic Inhibition
        # In high priority areas (like the mouth), we allow closer packing.
        # In low priority areas, we force drones to stay further apart.
        # priority 1.0 -> current_min = base_min_dist * 0.5 (Tighter)
        # priority 0.1 -> current_min = base_min_dist * 1.4 (Spread out)
        dynamic_min_dist = min_dist * (1.5 - priority)
        
        is_too_close = False
        for px, py in sampled_points:
            dist = np.sqrt((rx - px)**2 + (ry - py)**2)
            if dist < dynamic_min_dist:
                is_too_close = True
                break
        
        if not is_too_close:
            sampled_points.append((rx, ry))
            
    return sampled_points