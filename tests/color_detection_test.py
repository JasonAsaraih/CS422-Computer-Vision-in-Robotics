# Hannah Gray
# Version 1.4 – Non-Interactive Color Detection + Proper Red Detection

import cv2
import numpy as np

# HSV bounds for BLUE
lowerBlue = np.array([80, 120, 20])
upperBlue = np.array([130, 255, 255])

# HSV bounds for RED (needs two ranges!)
lowerRed1 = np.array([0, 120, 20])
upperRed1 = np.array([10, 255, 255])

lowerRed2 = np.array([170, 120, 20])
upperRed2 = np.array([180, 255, 255])

# Load test image
img = cv2.imread("test.jpeg")
if img is None:
    print("Error: test.jpeg not found!")
    exit()

# Convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# BLUE mask
maskBlue = cv2.inRange(hsv, lowerBlue, upperBlue)

# RED mask (two halves of hue circle)
maskRed1 = cv2.inRange(hsv, lowerRed1, upperRed1)
maskRed2 = cv2.inRange(hsv, lowerRed2, upperRed2)
maskRed = maskRed1 + maskRed2

# ------------------------------
# Draw BLUE bounding boxes
# ------------------------------
contoursB, _ = cv2.findContours(maskBlue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contoursB:
    if cv2.contourArea(contour) > 200:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)  # BLUE BOX
        cv2.putText(img, "BLUE", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# ------------------------------
# Draw RED bounding boxes
# ------------------------------
contoursR, _ = cv2.findContours(maskRed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contoursR:
    if cv2.contourArea(contour) > 200:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)  # RED BOX
        cv2.putText(img, "RED", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# ------------------------------
# Save the final result only
# ------------------------------
output_path = "color_detection_result.jpg"
cv2.imwrite(output_path, img)

print(f"Saved bounding box result to: {output_path}")
