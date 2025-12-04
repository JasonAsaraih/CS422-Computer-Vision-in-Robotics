import os
import random
import uuid
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

# =======================
# CONFIGURATION
# =======================
NUM_IMAGES_TO_GENERATE = 1000
OUTPUT_DIR = "Data"
IMAGE_SIZE = (150, 150)

MIN_BALLS = 1
MAX_BALLS = 5
MIN_SCALE = 0.05
MAX_SCALE = 0.5


# =======================
# PATHS
# =======================
BACKGROUND_DIR = "Data/Backgrounds"
BLUE_DIR = "Data/Blue Balls"
RED_DIR = "Data/Red Balls"

os.makedirs(f"{OUTPUT_DIR}/images/train", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels/train", exist_ok=True)

# =======================
# LOAD FILE LISTS
# =======================
background_files = [os.path.join(BACKGROUND_DIR, f) for f in os.listdir(BACKGROUND_DIR)]
blue_files = [os.path.join(BLUE_DIR, f) for f in os.listdir(BLUE_DIR)]
red_files = [os.path.join(RED_DIR, f) for f in os.listdir(RED_DIR)]


# =======================
# AUGMENTATION
# =======================
def random_augment(img):
    if random.random() < 0.5:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.7, 1.3))

    if random.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.2)))

    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    return img


# =======================
# BALL RESIZE
# =======================
def resize_ball(img):
    orig_w, orig_h = img.size
    scale = random.uniform(MIN_SCALE, MAX_SCALE)
    new_w = max(5, int(orig_w * scale))
    new_h = max(5, int(orig_h * scale))

    max_w, max_h = IMAGE_SIZE
    if new_w >= max_w or new_h >= max_h:
        scale_factor = min(max_w / orig_w, max_h / orig_h) * 0.8
        new_w = int(orig_w * scale_factor)
        new_h = int(orig_h * scale_factor)

    return img.resize((new_w, new_h), Image.LANCZOS)


# =======================
# YOLO LABEL FORMATTER (with clipping)
# =======================
def yolo_format_clip(xmin, ymin, xmax, ymax, img_w, img_h, class_id):
    # Clip to image boundaries
    xmin = max(0, min(img_w, xmin))
    xmax = max(0, min(img_w, xmax))
    ymin = max(0, min(img_h, ymin))
    ymax = max(0, min(img_h, ymax))

    # If completely offscreen skip it
    if xmax <= xmin or ymax <= ymin:
        return None

    x_center = (xmin + xmax) / 2 / img_w
    y_center = (ymin + ymax) / 2 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h

    return f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"


# =======================
# MAIN LOOP
# =======================
print("Generating synthetic dataset...")

for i in range(NUM_IMAGES_TO_GENERATE):

    bg_path = random.choice(background_files)
    bg = Image.open(bg_path).convert("RGB")
    bg = bg.resize(IMAGE_SIZE)
    bg = random_augment(bg)

    img_w, img_h = IMAGE_SIZE
    label_lines = []

    num_balls = random.randint(MIN_BALLS, MAX_BALLS)

    for _ in range(num_balls):

        class_type = random.choice(["blue", "red"])
        if class_type == "blue":
            ball_path = random.choice(blue_files)
            class_id = 0
        else:
            ball_path = random.choice(red_files)
            class_id = 1

        ball = Image.open(ball_path).convert("RGBA")
        ball = resize_ball(ball)
        ball = random_augment(ball)

        bw, bh = ball.size

        # Safety shrink if needed
        if bw >= img_w or bh >= img_h:
            shrink = min(img_w / bw, img_h / bh) * 0.8
            bw = int(bw * shrink)
            bh = int(bh * shrink)
            ball = ball.resize((bw, bh), Image.LANCZOS)

        # ============================================
        # NEW: Allow up to half of ball to be cut off
        # ============================================
        x = random.randint(-bw // 2, img_w - bw // 2)
        y = random.randint(-bh // 2, img_h - bh // 2)

        # Paste ball using alpha mask
        bg.paste(ball, (x, y), ball)

        # Bounding box BEFORE clipping
        xmin = x
        ymin = y
        xmax = x + bw
        ymax = y + bh

        yolo_line = yolo_format_clip(xmin, ymin, xmax, ymax, img_w, img_h, class_id)
        if yolo_line:
            label_lines.append(yolo_line)

    # Save outputs
    uid = uuid.uuid4().hex
    img_path = f"{OUTPUT_DIR}/images/train/{uid}.jpg"
    ann_path = f"{OUTPUT_DIR}/labels/train/{uid}.txt"

    bg.save(img_path)

    with open(ann_path, "w") as f:
        for line in label_lines:
            f.write(line + "\n")

    print(f"Generated {i+1}/{NUM_IMAGES_TO_GENERATE} images...")

print("Done! Synthetic YOLO dataset created successfully.")
