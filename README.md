# Robot & Game-Piece Detection System  
### **Hybrid Computer Vision + Deep Learning Pipeline (YOLOv8 + HSV + Edge Clustering)**

This repository contains a complete system for detecting **colored balls**, **robot edges**, and **clutter objects** in a robotics-competition environment. It combines:

- **YOLOv8 deep-learning detection**
- **HSV-based color segmentation**
- **Canny + morphological edge clustering**
- **Synthetic dataset generation for model training**

The project is designed to work with competition robots, where robots vary drastically in appearance and lighting is inconsistent.  
This pipeline provides robust multi-stage detection with redundancy across modules.

---

## Repository Overview

This project includes **7 Python scripts**:

### 1. `test.py` — YOLOv8 Inference Demo  
Loads a trained YOLO model and runs inference on a test image.

### 2. `train.py` — YOLOv8 Training Script  
Trains YOLOv8n on a dataset defined in `data.yaml`.

### 3. `edge_detection.py` — Partial-Circle + Edge-Based Detection  
Extracts edges, detects red/blue ball contours, and identifies clutter regions via contour clustering.

### 4. `data_generator.py` — Synthetic YOLO Dataset Generator  
Creates **automatically labeled, highly diverse synthetic training data** using random backgrounds, scale variation, occlusion, and augmentation.  
Outputs YOLO `.jpg` + `.txt` pairs.

### 5. `color_detection.py` — HSV Color-Based Detection  
Detects red and blue objects using dual-range red masking and contour filtering.

### 6. `Integration.py` — Full Combined Pipeline  
Integrates:  
✔ YOLOv8 detections  
✔ HSV detections  
✔ Edge suppression inside detected boxes  
✔ Edge clustering for clutter detection  
Outputs a complete annotated result image.

### 7. `main.py` — Template Program  
Example starter file from the IDE; included for completeness.

---

## Features

### ** Deep Learning (YOLOv8)**  
- Trained using custom synthetic dataset  
- Robust detection under varied conditions  
- Fast inference

### ** Classical Computer Vision**  
- HSV segmentation for red/blue  
- Canny-based edge extraction  
- Contour-based circle/partial-circle detection  
- Morphological clutter clustering

### ** Hybrid Multi-Stage Filtering**  
Each component improves the others:  
- YOLO + HSV boxes **suppress edges** inside them  
- Remaining edges identify robot/clutter  
- Geometric checks filter false positives

### ** Synthetic Dataset Pipeline**  
- Auto-generated labeled training images  
- Scalable to thousands of samples  
- Supports occlusions, random placement, augmentation  
- YOLO-format labels generated automatically

