import cv2
import numpy as np
import glob
import os
import shutil
from tqdm import tqdm

def resize_gray(image, size=(320, 240)):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def compute_optical_flow_score(img1, img2):
    flow = cv2.calcOpticalFlowFarneback(img1, img2, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return np.mean(magnitude)

# Define your starting image
main_image_filename = 'main.png'

# Load and resize all images in memory
image_paths = sorted(glob.glob('./*.png'))
print(f"Total images: {len(image_paths)}")

# Make sure main.png is in the list
if main_image_filename not in [os.path.basename(p) for p in image_paths]:
    raise FileNotFoundError(f"'{main_image_filename}' not found in ./")

# Map paths to index
basename_to_index = {os.path.basename(p): i for i, p in enumerate(image_paths)}
start_index = basename_to_index[main_image_filename]

# Load all grayscale, resized images
images_gray = [resize_gray(cv2.imread(p, cv2.IMREAD_GRAYSCALE)) for p in tqdm(image_paths, desc="Loading and resizing")]

# Prepare output
output_dir = "greedy_sorted_images"
os.makedirs(output_dir, exist_ok=True)

# Initialize chain with main.png
remaining_indices = set(range(len(images_gray)))
remaining_indices.remove(start_index)

chain = [start_index]
current_index = start_index

# Greedy chaining loop
for step in tqdm(range(len(images_gray) - 1), desc="Building chain"):
    current_image = images_gray[current_index]
    
    best_score = float('inf')
    best_index = None

    for i in remaining_indices:
        score = compute_optical_flow_score(current_image, images_gray[i])
        if score < best_score:
            best_score = score
            best_index = i

    chain.append(best_index)
    remaining_indices.remove(best_index)
    current_index = best_index

# Save the sorted chain
for rank, index in enumerate(chain):
    filename = f"{rank:04d}_{os.path.basename(image_paths[index])}"
    shutil.copy(image_paths[index], os.path.join(output_dir, filename))

print("Done! Images saved in greedy optical-flow-sorted order, starting from 'main.png'.")
