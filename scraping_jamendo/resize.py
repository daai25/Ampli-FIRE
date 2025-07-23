from PIL import Image
import os

# Input and output folder paths
input_folder = "spectrograms"
output_folder = "spectrograms_64"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Resize each image
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)

        # Resize image to 32x32
        resized_img = img.resize((64, 64), Image.LANCZOS)

        # Save resized image
        output_path = os.path.join(output_folder, filename)
        resized_img.save(output_path)

        print(f"Resized: {filename}")