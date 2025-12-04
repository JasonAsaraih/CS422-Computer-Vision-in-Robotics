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
    circle_circumference = 2 * np.pi * radius

    ratio = perimeter / circle_circumference
    return ratio >= min_ratio and radius > 0


# -------------------------------------------------------------
# Mask + edges for a color range
# -------------------------------------------------------------
def get_mask_and_edges(hsv, lower, upper, proc_frame, height):
    mask = cv2.inRange(hsv, lower, upper)

    # Morphology kernel = 1% of screen height
    kernel_size = int(height * 0.01)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernel_size, kernel_size))

    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    # Apply Otsu threshold on masked grayscale
    gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=cleaned)

    otsu_val, _ = cv2.threshold(masked_gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    low = max(10, int(otsu_val * 0.5))
    high = min(255, int(otsu_val * 1.5))

    edges = cv2.Canny(masked_gray, low, high)
    return cleaned, edges


# -------------------------------------------------------------
# Detect yellow clutter-box regions (improved threshold)
# -------------------------------------------------------------
def detect_clutter_boxes(gray_edges, height, width):
    # Downsample & blur to analyze density
    small = cv2.resize(gray_edges, (width // 4, height // 4))
    small = cv2.GaussianBlur(small, (11, 11), 0)

    # Threshold for "very cluttered" regions
    # (much higher threshold to require LOTS of edges)
    _, thresh = cv2.threshold(small, 65, 255, cv2.THRESH_BINARY)

    # Morph to merge clutter patches
    kernel = np.ones((7, 7), np.uint8)
    merged = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find clutter contours
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:   # require large region (less sensitive)
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Scale back up to original resolution
        boxes.append((
            x * 4,
            y * 4,
            w * 4,
            h * 4
        ))

    return boxes


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)   # your webcam

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    # Red HSV ranges
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Blue HSV range
    lower_blue = np.array([100, 150, 70])
    upper_blue = np.array([140, 255, 255])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]

        # Top cutoff %
        cutoff = int(height * 0.25)

        # Processing frame (top is black)
        proc_frame = frame.copy()
        proc_frame[:cutoff, :] = 0   # ensures not detected as red/blue

        # Display frame (top painted bright red)
        display_frame = frame.copy()
        display_frame[:cutoff, :] = (0, 0, 255)

        hsv = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2HSV)

        # --------------------
        # Red mask & edges
        # --------------------
        red_mask_1, red_edges_1 = get_mask_and_edges(hsv, lower_red1, upper_red1, proc_frame, height)
        red_mask_2, red_edges_2 = get_mask_and_edges(hsv, lower_red2, upper_red2, proc_frame, height)
        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
        red_edges = cv2.bitwise_or(red_edges_1, red_edges_2)

        # --------------------
        # Blue mask & edges
        # --------------------
        blue_mask, blue_edges = get_mask_and_edges(hsv, lower_blue, upper_blue, proc_frame, height)

        # --------------------
        # General edges (green)
        # --------------------
        gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
        general_edges = cv2.Canny(gray, 60, 150)

        # Draw general edges in green
        overlay = display_frame.copy()
        overlay[general_edges > 0] = (0, 255, 0)

        # Circle size constraints
        min_radius = int(height * 0.02)
        max_radius = int(height * 0.30)

        # --------------------
        # Draw RED circles
        #--------------------
        contours_red, _ = cv2.findContours(red_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_red:
            if cv2.contourArea(cnt) < 100:
                continue
            if is_partial_circle(cnt, 0.4):
                (x, y), r = cv2.minEnclosingCircle(cnt)
                r = int(r)
                if min_radius <= r <= max_radius:
                    cv2.circle(overlay, (int(x), int(y)), r, (0, 0, 255), 2)

        # --------------------
        # Draw BLUE circles
        # --------------------
        contours_blue, _ = cv2.findContours(blue_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_blue:
            if cv2.contourArea(cnt) < 100:
                continue
            if is_partial_circle(cnt, 0.4):
                (x, y), r = cv2.minEnclosingCircle(cnt)
                r = int(r)
                if min_radius <= r <= max_radius:
                    cv2.circle(overlay, (int(x), int(y)), r, (255, 0, 0), 2)

        # --------------------
        # Detect clutter & draw YELLOW BOXES
        # --------------------
        clutter_boxes = detect_clutter_boxes(general_edges, height, width)
        for (x, y, w, h) in clutter_boxes:
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 255), 2)

        # --------------------
        # Show outputs
        # --------------------
        cv2.imshow("Overlay Output", overlay)
        cv2.imshow("Red Mask", red_mask)
        cv2.imshow("Blue Mask", blue_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# -------------------------------------------------------------
# RUN
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
