import os
from rembg import remove
from PIL import Image
import pillow_heif
import io

# Enable HEIC/HEIF support in Pillow
pillow_heif.register_heif_opener()

# --- Paths ---
input_dir = "Data/Field Elements/Blue Balls"
output_dir = "Data/Field Elements/Removed Background and Cropped Blue Balls"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Supported image extensions
valid_exts = (".png", ".jpg", ".jpeg", ".heic")

# Loop through all images in the input directory
for index, filename in enumerate(os.listdir(input_dir)):
    if filename.lower().endswith(valid_exts):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"Image_{index}.png")

        print(f"Processing: {filename}")

        # --- Open the image (supports HEIC) ---
        with Image.open(input_path) as img:
            img = img.convert("RGBA")

            # Save Pillow image into a memory buffer as PNG
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            input_bytes = buffer.getvalue()

            # --- Remove background using rembg ---
            output_bytes = remove(input_bytes)

            # Open result back into Pillow
            result_img = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

            # --- Auto-crop transparent borders ---
            # Get the alpha channel and find the bounding box of non-zero alpha
            alpha = result_img.getchannel("A")
            bbox = alpha.getbbox()  # returns (left, upper, right, lower)
            if bbox:
                result_img = result_img.crop(bbox)

            # --- Save cropped image ---
            result_img.save(output_path, format="PNG")

        print(f"Saved: {output_path}")

print("All images processed!")
