import cv2
import numpy as np

# -------------------------------------------------------------
# Check if contour matches partial circle
# -------------------------------------------------------------
def is_partial_circle(contour, min_ratio=0.4):
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False

    (x, y), radius = cv2.minEnclosingCircle(contour)
    circle_circ = 2 * np.pi * radius

    ratio = perimeter / circle_circ
    return ratio >= min_ratio and radius > 0


# -------------------------------------------------------------
# Mask + edges for a color range
# -------------------------------------------------------------
def get_mask_and_edges(hsv, lower, upper, proc_frame, height):
    mask = cv2.inRange(hsv, lower, upper)

    # Morphology kernel
    kernel_size = int(height * 0.01)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernel_size, kernel_size))

    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=cleaned)

    otsu_val, _ = cv2.threshold(masked_gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    low = max(10, int(otsu_val * 0.5))
    high = min(255, int(otsu_val * 1.5))

    edges = cv2.Canny(masked_gray, low, high)
    return cleaned, edges


# -------------------------------------------------------------
# Detect clutter boxes
# -------------------------------------------------------------
def detect_clutter_boxes(gray_edges, height, width):
    small = cv2.resize(gray_edges, (width // 4, height // 4))
    small = cv2.GaussianBlur(small, (11, 11), 0)

    _, thresh = cv2.threshold(small, 65, 255, cv2.THRESH_BINARY)

    kernel = np.ones((7, 7), np.uint8)
    merged = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 500:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x * 4, y * 4, w * 4, h * 4))

    return boxes


# -------------------------------------------------------------
# MAIN — Processes a single test image instead of webcam
# -------------------------------------------------------------
def main():

    # Load your test image
    frame = cv2.imread("test.jpeg")
    if frame is None:
        print("ERROR: test.jpeg not found!")
        return

    height, width = frame.shape[:2]

    # Remove top clutter by blacking it out
    cutoff = int(height * 0.25)
    proc_frame = frame.copy()
    proc_frame[:cutoff] = 0

    # Output image
    overlay = frame.copy()
    overlay[:cutoff] = (0, 0, 255)

    hsv = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2HSV)

    # RED ranges
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # BLUE range
    lower_blue = np.array([100, 150, 70])
    upper_blue = np.array([140, 255, 255])

    # ---------------------------------------------------------
    # RED Masks + Edges
    # ---------------------------------------------------------
    red_mask_1, red_edges_1 = get_mask_and_edges(hsv, lower_red1, upper_red1, proc_frame, height)
    red_mask_2, red_edges_2 = get_mask_and_edges(hsv, lower_red2, upper_red2, proc_frame, height)
    red_edges = cv2.bitwise_or(red_edges_1, red_edges_2)

    # ---------------------------------------------------------
    # BLUE Mask + Edges
    # ---------------------------------------------------------
    blue_mask, blue_edges = get_mask_and_edges(hsv, lower_blue, upper_blue, proc_frame, height)

    # ---------------------------------------------------------
    # General Edges — NOW THICKER!
    # ---------------------------------------------------------
    gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
    general_edges = cv2.Canny(gray, 60, 150)

    # Thicken general edges using dilation
    kernel = np.ones((3, 3), np.uint8)
    thick_edges = cv2.dilate(general_edges, kernel, iterations=2)

    # Draw thick edges in green
    overlay[thick_edges > 0] = (0, 255, 0)

    # Circle size limits
    min_radius = int(height * 0.02)
    max_radius = int(height * 0.30)

    # ---------------------------------------------------------
    # Draw RED circles
    # ---------------------------------------------------------
    contours_red, _ = cv2.findContours(red_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_red:
        if cv2.contourArea(cnt) < 100:
            continue
        if is_partial_circle(cnt):
            (x, y), r = cv2.minEnclosingCircle(cnt)
            r = int(r)
            if min_radius <= r <= max_radius:
                cv2.circle(overlay, (int(x), int(y)), r, (0, 0, 255), 2)

    # ---------------------------------------------------------
    # Draw BLUE circles
    # ---------------------------------------------------------
    contours_blue, _ = cv2.findContours(blue_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_blue:
        if cv2.contourArea(cnt) < 100:
            continue
        if is_partial_circle(cnt):
            (x, y), r = cv2.minEnclosingCircle(cnt)
            r = int(r)
            if min_radius <= r <= max_radius:
                cv2.circle(overlay, (int(x), int(y)), r, (255, 0, 0), 2)

    # ---------------------------------------------------------
    # Draw YELLOW clutter boxes
    # ---------------------------------------------------------
    clutter_boxes = detect_clutter_boxes(general_edges, height, width)
    for (x, y, w, h) in clutter_boxes:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # ---------------------------------------------------------
    # SAVE ONLY — No display
    # ---------------------------------------------------------
    output_path = "edge_detection_output.jpg"
    cv2.imwrite(output_path, overlay)
    print(f"Saved processed image to: {output_path}")


# -------------------------------------------------------------
# RUN
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
