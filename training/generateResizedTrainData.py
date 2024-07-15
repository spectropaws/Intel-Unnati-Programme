import os
from PIL import Image, ImageOps
import shutil
import random

# Function to resize an image to match the target size by cropping
def resize_image(image_path, output_path, target_size):
    img = Image.open(image_path)
    # Resize by cropping to match target size
    img = ImageOps.fit(img, target_size, method=Image.LANCZOS, bleed=0.0, centering=(0.5, 0.5))
    img.save(output_path)

# Function to pixelate an image with a specified factor
def pixelate_image(image_path, output_path, pixelation_factor):
    img = Image.open(image_path)
    # Resize by pixelation factor
    small_img = img.resize((img.width // pixelation_factor, img.height // pixelation_factor), resample=Image.BILINEAR)
    # Resize back to original size to achieve pixelation effect
    pixelated = small_img.resize(img.size, Image.NEAREST)
    pixelated.save(output_path)

# Path to original images folder
original_images_folder = "images/"
# Path to data folder
data_folder = "data/"

# Create necessary folders if they don't exist
os.makedirs(data_folder, exist_ok=True)
for folder in ['train/X', 'train/y', 'test/X', 'test/y']:
    os.makedirs(os.path.join(data_folder, folder), exist_ok=True)

# List all images in the original images folder
image_files = os.listdir(original_images_folder)
random.shuffle(image_files)  # Shuffle the list of images

# Calculate the split based on 80% train and 20% test
split_ratio = 0.8
split_index = int(len(image_files) * split_ratio)

# Target size for resizing
target_size = (256, 256)
# Pixelation factor
pixelation_factor = 3  # Adjust as needed

# Process each image
for i, image_file in enumerate(image_files, start=1):
    original_image_path = os.path.join(original_images_folder, image_file)
    resized_y_image_path = os.path.join(data_folder, f"y/{i}.jpg")
    resized_x_image_path = os.path.join(data_folder, f"X/{i}.jpg")

    # Resize and save y (original) image
    resize_image(original_image_path, resized_y_image_path, target_size)

    # Create and save pixelated version in X
    pixelate_image(resized_y_image_path, resized_x_image_path, pixelation_factor)

    if i <= split_index:
        # Move to train folder
        shutil.move(resized_y_image_path, os.path.join(data_folder, "train/y"))
        shutil.move(resized_x_image_path, os.path.join(data_folder, "train/X"))
    else:
        # Move to test folder
        shutil.move(resized_y_image_path, os.path.join(data_folder, "test/y"))
        shutil.move(resized_x_image_path, os.path.join(data_folder, "test/X"))

print("Image processing and folder structure creation complete.")
