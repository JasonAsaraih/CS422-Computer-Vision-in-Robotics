from ultralytics import YOLO
import cv2

model = YOLO("C:\\runs\\detect\\ball_detector4\\weights\\best.pt")
print(model.names)

img = cv2.imread("test.jpeg")
results = model(img, conf=0.10)

annotated = results[0].plot()

# Save the annotated result
output_path = "result.jpg"
cv2.imwrite(output_path, annotated)
print(f"Saved: {output_path}")
