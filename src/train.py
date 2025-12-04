from ultralytics import YOLO

# ---------------------------------------------------------
# Load a base YOLO model (yolov8n is good for 150x150 data)
# ---------------------------------------------------------
model = YOLO("yolov8n.pt")   # can switch to yolov8s.pt later

# ---------------------------------------------------------
# Train
# ---------------------------------------------------------
model.train(
    task="detect",
    data="data.yaml",
    epochs=20,
    imgsz=150,
    batch=16,
    name="ball_detector",
    workers=4,
)
