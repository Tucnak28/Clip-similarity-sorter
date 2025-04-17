from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import os
import shutil

# Load the OpenAI CLIP Model
print('Loading CLIP Model...')
model = SentenceTransformer('clip-ViT-B-32')

# Gather image file names
image_names = list(glob.glob('./*.png'))
print("Images:", len(image_names))

# Set the main image file
main_image_path = './main.png'  # Replace with the path to your main image
main_image_name = os.path.basename(main_image_path)

# Encode the main image
main_image = Image.open(main_image_path)
encoded_main_image = model.encode([main_image], convert_to_tensor=True)

# Encode all images
encoded_images = model.encode([Image.open(filepath) for filepath in image_names], 
                              batch_size=128, 
                              convert_to_tensor=True, 
                              show_progress_bar=True)

# Compute similarity scores between the main image and all other images
similarity_scores = []
for idx, encoded_image in enumerate(encoded_images):
    score = util.pytorch_cos_sim(encoded_main_image, encoded_image).item()
    similarity_scores.append((score, idx))

# Sort the images by similarity score (higher score means more similar)
sorted_scores = sorted(similarity_scores, key=lambda x: x[0], reverse=True)

# Create a new directory for sorted images
output_dir = "sorted_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Copy the images to the new directory with names based on similarity score
for idx, (score, image_id) in enumerate(sorted_scores):
    # Define the new filename based on the similarity score
    score_str = f"{score:.3f}".replace('0.', '0,') #replace 0. with the score, being 1000 as the highest
    new_name = f"{score_str}_{os.path.basename(image_names[image_id])}"
    
    # Copy the image to the output directory with the new name
    shutil.copy(image_names[image_id], os.path.join(output_dir, new_name))

    print(f"Rank {idx+1}: Score {score:.3f}")
    print(f"Image copied as: {new_name}")

print("Images have been sorted, renamed, and copied to the 'sorted_images' folder.")
