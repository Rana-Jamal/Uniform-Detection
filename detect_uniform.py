# detect_uniform.py

import os
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# === CONFIGURATION ===
MODEL_PATH           = "Helping Files/best.pt"
CLASSES_PATH         = "Helping Files/classes.txt"
CONFIDENCE_THRESHOLD = 0.40   # only keep detections ≥ this score
# ======================

def load_classes(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def uniform_detection_single(pid):
    """Process uniform detection for a single person"""
    # Parse BakeryID from pid
    bakery_id = pid[-1]

    # Set dynamic directories
    base_dir = os.path.join("Bakeries", bakery_id, "Captured")
    frames_dir = os.path.join(base_dir, "frames")
    uniform_bbox_dir = os.path.join(base_dir, "uniform_bbox")
    os.makedirs(uniform_bbox_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)  # Ensure frames dir exists

    today = datetime.now().strftime("%Y-%m-%d")
    out_file = os.path.join(uniform_bbox_dir, f"{today}.txt")

    class_names = load_classes(CLASSES_PATH)
    model = YOLO(MODEL_PATH)

    # Process the specific person's frame
    fname = f"{pid}_{today}.jpg"
    img_path = os.path.join(frames_dir, fname)
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"⚠️ Could not load {img_path}")
        return

    # run with custom confidence threshold
    result = model(img, conf=CONFIDENCE_THRESHOLD)[0]
    bbox_entries = []

    # **Handle oriented boxes (OBB) if present**
    if getattr(result, "obb", None) and result.obb.data is not None:
        obb_array = result.obb.data.cpu().numpy()
        for row in obb_array:
            cid, conf = int(row[6]), float(row[5])
            if conf >= CONFIDENCE_THRESHOLD and 0 <= cid < len(class_names):
                x_center, y_center, width, height, angle = row[0], row[1], row[2], row[3], row[4]
                class_name = class_names[cid]
                bbox_entries.append(
                    f"{class_name}({conf:.2f}):xc={int(x_center)},yc={int(y_center)},w={int(width)},h={int(height)},angle={angle:.2f}"
                )

    # Fallback: standard boxes
    elif getattr(result, "boxes", None) and getattr(result.boxes, "cls", None) is not None:
        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        cls_idxs = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        
        for (x1, y1, x2, y2), cid, conf in zip(boxes, cls_idxs, confs):
            if conf >= CONFIDENCE_THRESHOLD and 0 <= cid < len(class_names):
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                class_name = class_names[cid]
                bbox_entries.append(f"{class_name}({conf:.2f}):x={x1},y={y1},w={x2-x1},h={y2-y1}")

    # Save bbox info
    bbox_str = " | ".join(bbox_entries) if bbox_entries else "none"
    line = f"{pid}_{today} : {bbox_str}"

    # Thread-safe file writing
    with open(out_file, "a") as f:
        f.write(line + "\n")

    print(f"✅ Processed uniform detection for {pid}: {len(bbox_entries)} uniforms detected")