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
print("Images found:", len(image_names))

# Define the prompt
prompt = "dog"

# Encode the prompt
encoded_prompt = model.encode([prompt], convert_to_tensor=True)

# Encode all images
encoded_images = model.encode([Image.open(filepath) for filepath in image_names], 
                              batch_size=128, 
                              convert_to_tensor=True, 
                              show_progress_bar=True)

# Compute similarity scores between the prompt and all images
similarity_scores = util.pytorch_cos_sim(encoded_prompt, encoded_images)[0].cpu().numpy()

# Pair each image with its similarity score
image_scores = [(score, idx) for idx, score in enumerate(similarity_scores)]

# Sort the images by similarity score (higher score means more relevant)
sorted_image_scores = sorted(image_scores, key=lambda x: x[0], reverse=True)

# Select the top 20 most relevant images
top_20_images = sorted_image_scores[:20]

# Create a new directory for the top 20 images
output_dir = "top_20_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Copy the images to the new directory with names based on similarity score
for score, image_id in top_20_images:
    # Define the new filename based on the similarity score
    score_percentage = f"{score * 100:.2f}".replace('0.', '0,')
    new_name = f"{score_percentage}.jpg"
    
    # Copy the image to the output directory with the new name
    shutil.copy(image_names[image_id], os.path.join(output_dir, new_name))

    print(f"Image '{image_names[image_id]}' copied as '{new_name}' with score: {score_percentage}%")

print("Top 20 images have been saved in the 'top_20_images' folder.")
