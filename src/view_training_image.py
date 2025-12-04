import os
import random
import numpy as np
from PIL import Image, ImageDraw

# --- Paths ---
image_dir = "Data/Training Data/10-24-25_200imgs"
label_dir = os.path.join(image_dir, "labels")

# --- Parameters ---
max_balls = 15  # must match your training data

# --- Choose a random image ---
images = [f for f in os.listdir(image_dir) if f.lower().endswith(".png")]
if not images:
    raise ValueError("No images found in the directory.")

img_name = random.choice(images)
img_path = os.path.join(image_dir, img_name)
label_path = os.path.join(label_dir, img_name.replace(".png", ".npy"))

# --- Load image and labels ---
img = Image.open(img_path).convert("RGB")
label_vector = np.load(label_path)

print(f"Displaying {img_name}")
print("Full training vector:", label_vector)

# --- Draw bounding boxes ---
draw = ImageDraw.Draw(img)
for i in range(max_balls):
    start = i * 5
    color_class, x_center, y_center, width_norm, height_norm = label_vector[start:start + 5]

    if color_class == 0:
        continue  # skip empty slots

    # Convert normalized coords to pixel values
    x_center_px = x_center * img.width
    y_center_px = y_center * img.height
    width_px = width_norm * img.width
    height_px = height_norm * img.height

    # Compute box coordinates
    left = x_center_px - width_px / 2
    top = y_center_px - height_px / 2
    right = x_center_px + width_px / 2
    bottom = y_center_px + height_px / 2

    # Choose box color: red or blue
    box_color = "red" if color_class == 1 else "blue"

    # Draw rectangle
    draw.rectangle([left, top, right, bottom], outline=box_color, width=2)

# --- Show image ---
img.show()
