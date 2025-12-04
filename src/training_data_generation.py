import os
import random
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

# --- Paths ---
background_dir = "Data/Background Images"
red_ball_dir = "Data/Field Elements/Removed Background and Cropped Red Balls"
blue_ball_dir = "Data/Field Elements/Removed Background and Cropped Blue Balls"
output_dir = "Data/Training Data/10-24-25_200imgs"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

# --- Parameters ---
num_images = 200
ball_prob = 0.7
max_balls_per_image = 15
augment_prob = 0.5
max_offscreen_fraction = 0.75  # max % of ball allowed off screen

backgrounds = [os.path.join(background_dir, f) for f in os.listdir(background_dir)
               if f.lower().endswith((".jpg", ".png"))]
red_balls = [os.path.join(red_ball_dir, f) for f in os.listdir(red_ball_dir) if f.lower().endswith(".png")]
blue_balls = [os.path.join(blue_ball_dir, f) for f in os.listdir(blue_ball_dir) if f.lower().endswith(".png")]

def random_augment_background(bg):
    if random.random() < augment_prob:
        enhancer = ImageEnhance.Brightness(bg)
        bg = enhancer.enhance(random.uniform(0.7, 1.3))
    if random.random() < augment_prob:
        bg = bg.filter(ImageFilter.GaussianBlur(random.uniform(0, 1)))
    return bg

def random_augment_ball(ball):
    if random.random() < augment_prob:
        angle = random.uniform(-30, 30)
        ball = ball.rotate(angle, expand=True)
    if random.random() < augment_prob:
        enhancer = ImageEnhance.Brightness(ball)
        ball = enhancer.enhance(random.uniform(0.5, 1.5))
    return ball

for i in range(num_images):
    bg = Image.open(random.choice(backgrounds)).convert("RGBA")
    bg = random_augment_background(bg)
    ball_labels = []

    if random.random() < ball_prob:
        num_balls = random.randint(1, 5)
        for _ in range(num_balls):
            color_class = random.choice([1, 2])
            ball_list = red_balls if color_class == 1 else blue_balls
            ball = Image.open(random.choice(ball_list)).convert("RGBA")

            # Scale dynamically, min 5% of bg, max 30%
            min_scale_w = 0.05 * bg.width / ball.width
            min_scale_h = 0.05 * bg.height / ball.height
            min_scale = max(min_scale_w, min_scale_h)
            max_scale_w = bg.width / ball.width
            max_scale_h = bg.height / ball.height
            max_scale = min(max_scale_w, max_scale_h, 0.3)
            scale = random.uniform(min_scale, max_scale)
            new_size = (max(1, int(ball.width * scale)), max(1, int(ball.height * scale)))
            ball = ball.resize(new_size, Image.LANCZOS)

            ball = random_augment_ball(ball)

            # Allow up to 75% off-screen
            max_off_x = int(ball.width * max_offscreen_fraction)
            max_off_y = int(ball.height * max_offscreen_fraction)
            x = random.randint(-max_off_x, bg.width - ball.width + max_off_x)
            y = random.randint(-max_off_y, bg.height - ball.height + max_off_y)

            # Overlay ball
            bg.paste(ball, (x, y), ball)

            # YOLO-style bounding box (clip centers and sizes to [0,1])
            x_center = (x + ball.width / 2) / bg.width
            y_center = (y + ball.height / 2) / bg.height
            width_norm = ball.width / bg.width
            height_norm = ball.height / bg.height

            ball_labels.append([color_class, x_center, y_center, width_norm, height_norm])

    # Fill remaining slots
    while len(ball_labels) < max_balls_per_image:
        ball_labels.append([0, 0, 0, 0, 0])

    output_vector = np.array(ball_labels).flatten()

    out_path = os.path.join(output_dir, f"synthetic_{i:03d}.png")
    bg.convert("RGB").save(out_path)

    label_path = os.path.join(output_dir, "labels", f"synthetic_{i:03d}.npy")
    np.save(label_path, output_vector)

    print(f"Saved {out_path} with {len(ball_labels)} ball slots, vector length: {len(output_vector)}")

print("Synthetic dataset generation complete!")
