# # detect_person.py

# import os
# import cv2
# import numpy as np
# from datetime import datetime
# from ultralytics import YOLO
# import sqlite3

# # === CONFIGURATION ===
# MODEL_PATH     = "Helping Files/yolo11s.pt"
# UNIFORM_MODEL_PATH = "Helping Files/best.pt"
# CLASSES_PATH   = "Helping Files/classes.txt"
# CONFIDENCE_THRESHOLD = 0.40   # only keep detections ≥ this score

# # Define uniform classes - add your actual classes here
# ACTUAL_UNIFORM_CLASSES = {
#     "Manager Uniform",
#     "Salesman Uniform 1", 
#     "Salesman Uniform 2",
#     "Sweeper Uniform"
# }

# # Classes that should be excluded when actual uniforms are present
# EXCLUSION_CLASSES = {
#     "No Uniform"
# }
# # ======================

# def load_classes(path):
#     with open(path, "r") as f:
#         return [line.strip() for line in f if line.strip()]

# def get_class_colors(class_names):
#     """Generate consistent colors for each class"""
#     np.random.seed(42)  # Fixed seed for consistent colors
#     colors = {}
#     for i, class_name in enumerate(class_names):
#         colors[class_name] = tuple(map(int, np.random.randint(0, 255, 3)))
#     return colors

# def parse_loc_file(path, target_pid):
#     """
#     Reads lines like "ahmad_2025-08-06 : x=123,y=45, w=100, h=100"
#     Returns (x, y, w, h) for target_pid or None
#     """
#     face_bbox = None
#     with open(path, "r") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             pid_date, coords = line.split(":", 1)
#             current_pid = pid_date.strip().rsplit("_", 1)[0]
#             if current_pid == target_pid:
#                 parts = coords.replace(" ", "").split(",")
#                 x = int(parts[0].split("=")[1])
#                 y = int(parts[1].split("=")[1])
#                 w = int(parts[2].split("=")[1])
#                 h = int(parts[3].split("=")[1])
#                 face_bbox = (x, y, w, h)
#                 break
#     return face_bbox

# def parse_uniform_bbox_file(path, target_pid):
#     """
#     Reads lines like "ahmad_2025-08-06 : class1(0.95):x=10,y=20,w=50,h=60 | class2(0.87):x=70,y=80,w=30,h=40"
#     Returns list [(class_name, conf, x, y, w, h), ...] for target_pid
#     """
#     uniforms = []
#     with open(path, "r") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             pid_date, bbox_info = line.split(":", 1)
#             current_pid = pid_date.strip().rsplit("_", 1)[0]
#             if current_pid == target_pid:
#                 if bbox_info.strip() != "none":
#                     uniform_parts = bbox_info.split(" | ")
#                     for part in uniform_parts:
#                         part = part.strip()
#                         if not part:
#                             continue
                        
#                         class_conf, coords = part.split(":", 1)
#                         class_name = class_conf.split("(")[0]
#                         conf = float(class_conf.split("(")[1].split(")")[0])
                        
#                         coord_parts = coords.split(",")
#                         x = int(coord_parts[0].split("=")[1])
#                         y = int(coord_parts[1].split("=")[1])
#                         w = int(coord_parts[2].split("=")[1])
#                         h = int(coord_parts[3].split("=")[1])
                        
#                         uniforms.append((class_name, conf, x, y, w, h))
#                 break
#     return uniforms

# def boxes_overlap(box1, box2):
#     """Check if two bounding boxes overlap"""
#     x1, y1, w1, h1 = box1
#     x2, y2, w2, h2 = box2
    
#     box1_x2, box1_y2 = x1 + w1, y1 + h1
#     box2_x2, box2_y2 = x2 + w2, y2 + h2
    
#     return not (box1_x2 <= x2 or box2_x2 <= x1 or box1_y2 <= y2 or box2_y2 <= y1)

# def calculate_overlap_area(box1, box2):
#     """Calculate the overlap area between two boxes"""
#     x1, y1, w1, h1 = box1
#     x2, y2, w2, h2 = box2
    
#     box1_x2, box1_y2 = x1 + w1, y1 + h1
#     box2_x2, box2_y2 = x2 + w2, y2 + h2
    
#     inter_x1 = max(x1, x2)
#     inter_y1 = max(y1, y2)
#     inter_x2 = min(box1_x2, box2_x2)
#     inter_y2 = min(box1_y2, box2_y2)
    
#     if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
#         return 0
    
#     return (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

# def filter_uniform_results(person_uniforms):
#     """
#     Filter uniform results based on business logic:
#     - If any actual uniform is detected, exclude "No Uniform" and similar exclusion classes
#     - If multiple uniform classes (from ACTUAL_UNIFORM_CLASSES) are detected, keep only the one with the highest confidence
#     - Keep non-uniform classes (e.g., Cap, No Cap) intact
    
#     Args:
#         person_uniforms: dict {class_name: confidence}
    
#     Returns:
#         dict: filtered uniform results
#     """
#     # Step 1: If any actual uniform is detected, exclude classes in EXCLUSION_CLASSES
#     has_actual_uniform = any(class_name in ACTUAL_UNIFORM_CLASSES for class_name in person_uniforms.keys())
    
#     if has_actual_uniform:
#         filtered_uniforms = {
#             class_name: conf for class_name, conf in person_uniforms.items() 
#             if class_name not in EXCLUSION_CLASSES
#         }
#     else:
#         filtered_uniforms = person_uniforms.copy()
    
#     # Step 2: If multiple uniform classes (from ACTUAL_UNIFORM_CLASSES) are present, keep only the one with highest confidence
#     uniform_classes = {k: v for k, v in filtered_uniforms.items() if k in ACTUAL_UNIFORM_CLASSES}
#     if len(uniform_classes) > 1:
#         # Keep the uniform class with the highest confidence
#         max_conf_uniform = max(uniform_classes.items(), key=lambda x: x[1])
#         # Retain non-uniform classes and the highest-confidence uniform class
#         filtered_uniforms = {
#             k: v for k, v in filtered_uniforms.items() 
#             if k not in ACTUAL_UNIFORM_CLASSES or k == max_conf_uniform[0]
#         }
    
#     return filtered_uniforms

# def person_detection_single(pid):
#     """Process person detection for a single person"""
#     # Parse BakeryID from pid
#     bakery_id = pid[-1]

#     # Set dynamic directories
#     base_dir = os.path.join("Bakeries", bakery_id, "Captured")
#     frames_dir = os.path.join(base_dir, "frames")
#     loc_info_dir = os.path.join(base_dir, "loc_info")
#     uniform_bbox_dir = os.path.join(base_dir, "uniform_bbox")
#     final_results_dir = os.path.join(base_dir, "final_results")
#     person_frames_dir = os.path.join(base_dir, "person_frame")
#     annotated_frames_dir = os.path.join(base_dir, "annotated_frames")
#     os.makedirs(final_results_dir, exist_ok=True)
#     os.makedirs(person_frames_dir, exist_ok=True)
#     os.makedirs(annotated_frames_dir, exist_ok=True)
#     os.makedirs(loc_info_dir, exist_ok=True)
#     os.makedirs(uniform_bbox_dir, exist_ok=True)
#     os.makedirs(frames_dir, exist_ok=True)

#     today = datetime.now().strftime("%Y-%m-%d")
#     loc_file = os.path.join(loc_info_dir, f"{today}.txt")
#     uniform_bbox_file = os.path.join(uniform_bbox_dir, f"{today}.txt")
    
#     # Get face bbox for this specific person
#     face_bbox = parse_loc_file(loc_file, pid)
#     if not face_bbox:
#         print(f"No face bbox found for {pid}")
#         return

#     fx, fy, fw, fh = face_bbox
    
#     # Get uniform detections for this specific person
#     uniform_detections = parse_uniform_bbox_file(uniform_bbox_file, pid)
    
#     model = YOLO(MODEL_PATH)
#     uniform_model = YOLO(UNIFORM_MODEL_PATH)
#     class_names = load_classes(CLASSES_PATH)
#     class_colors = get_class_colors(class_names)
    
#     # STEP 1: First run person detection on frame and save annotated frame
#     print(f"Step 1: Running person detection and saving annotated frame for {pid}...")
    
#     frame_path = os.path.join(frames_dir, f"{pid}_{today}.jpg")
#     img = cv2.imread(frame_path)
#     if img is None:
#         print(f"Could not load image: {frame_path}")
#         return

#     # Create annotated image copy
#     annotated_img = img.copy()

#     # Draw face bounding box (GREEN)
#     cv2.rectangle(annotated_img, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
#     cv2.putText(annotated_img, f"Face: {pid}", (fx, fy - 10),
#                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     # Run uniform detection and draw uniforms
#     uniform_result = uniform_model(img, conf=CONFIDENCE_THRESHOLD)[0]

#     # Handle oriented boxes (OBB) if present
#     if getattr(uniform_result, "obb", None) and uniform_result.obb.data is not None:
#         obb_array = uniform_result.obb.data.cpu().numpy()
#         for row in obb_array:
#             cid, conf = int(row[6]), float(row[5])
#             if conf >= CONFIDENCE_THRESHOLD and 0 <= cid < len(class_names):
#                 x_center, y_center, width, height, angle = row[0], row[1], row[2], row[3], row[4]
                
#                 rect = ((x_center, y_center), (width, height), np.degrees(angle))
#                 box = cv2.boxPoints(rect)
#                 box = box.astype(np.int32)

#                 class_name = class_names[cid]
#                 color = class_colors[class_name]
#                 cv2.drawContours(annotated_img, [box], 0, color, 2)
#                 cv2.putText(annotated_img, f"{class_name}({conf:.2f})",
#                            (int(x_center), int(y_center)),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     # Handle standard boxes
#     elif getattr(uniform_result, "boxes", None) and getattr(uniform_result.boxes, "cls", None) is not None:
#         boxes = uniform_result.boxes.xyxy.cpu().numpy()
#         cls_idxs = uniform_result.boxes.cls.cpu().numpy().astype(int)
#         confs = uniform_result.boxes.conf.cpu().numpy()
        
#         for (x1, y1, x2, y2), cid, conf in zip(boxes, cls_idxs, confs):
#             if conf >= CONFIDENCE_THRESHOLD and 0 <= cid < len(class_names):
#                 x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
#                 class_name = class_names[cid]
#                 color = class_colors[class_name]
#                 cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(annotated_img, f"{class_name}({conf:.2f})", (x1, y1-10),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     # Run person detection and draw ALL detected persons
#     results = model(img, classes=[0])  # only person class
#     person_boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes.xyxy) > 0 else []
    
#     # Draw ALL person bounding boxes (BLUE)
#     for i, person_box in enumerate(person_boxes):
#         x1, y1, x2, y2 = map(int, person_box)
#         cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
#         cv2.putText(annotated_img, f"Person_{i+1}", (x1, y1 - 10),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

#     # Save annotated frame
#     annotated_path = os.path.join(annotated_frames_dir, f"{pid}_{today}_annotated.jpg")
#     cv2.imwrite(annotated_path, annotated_img)
#     print(f"    Saved annotated frame: {annotated_path}")

#     # STEP 2: Now do the matching and save final results for {pid}...
#     print(f"Step 2: Matching face to person and saving final results for {pid}...")
    
#     results = model(img, classes=[0])  # only person class
#     boxes   = results[0].boxes.xyxy
#     classes = results[0].boxes.cls

#     # Face center point
#     cx = fx + fw / 2
#     cy = fy + fh / 2

#     person_found = False
#     for (x1, y1, x2, y2), cls in zip(boxes, classes):
#         x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        
#         # Check if face center is within person bounding box
#         if x1 <= cx <= x2 and y1 <= cy <= y2:
#             person_bbox = (x1, y1, x2-x1, y2-y1)  # Convert to x, y, w, h
            
#             # Crop and save the recognized person
#             person_crop = img[y1:y2, x1:x2]
#             person_frame_path = os.path.join(person_frames_dir, f"{pid}_{today}.jpg")
#             cv2.imwrite(person_frame_path, person_crop)
#             print(f"    Saved person crop: {person_frame_path}")
            
#             # Find overlapping uniforms for this person
#             person_uniforms = {}  # Use dict to store highest confidence for each class
#             for uniform_data in uniform_detections:
#                 class_name, conf, ux, uy, uw, uh = uniform_data
#                 uniform_bbox = (ux, uy, uw, uh)
                
#                 if boxes_overlap(person_bbox, uniform_bbox):
#                     overlap_area = calculate_overlap_area(person_bbox, uniform_bbox)
#                     uniform_area = uw * uh
#                     overlap_ratio = overlap_area / uniform_area if uniform_area > 0 else 0
                    
#                     if overlap_ratio > 0.1:  # At least 10% overlap
#                         if class_name not in person_uniforms or conf > person_uniforms[class_name]:
#                             person_uniforms[class_name] = conf
            
#             # Apply filtering logic to remove exclusion classes and keep highest confidence uniform
#             filtered_uniforms = filter_uniform_results(person_uniforms)
            
#             # Create result entry with filtered uniform classes
#             uniform_list = [f"{class_name}({conf:.2f})" for class_name, conf in filtered_uniforms.items()]
#             uniform_str = ", ".join(uniform_list) if uniform_list else "none"
#             result_line = f"{pid}_{today} : {uniform_str}"
            
#             # Thread-safe file writing to final_results.txt (optional, but kept as per original)
#             final_results_file = os.path.join(final_results_dir, f"{today}.txt")
#             with open(final_results_file, "a") as f:
#                 f.write(result_line + "\n")
            
#             print(f"Processed {pid}:")
#             print(f"    Raw uniforms detected: {person_uniforms}")
#             print(f"    Filtered uniforms: {uniform_str}")
            
#             # Integrate with SQLite DB
#             db_path = os.path.join("Bakeries", "Uniform_information.db")
#             os.makedirs(os.path.dirname(db_path), exist_ok=True)
#             conn = sqlite3.connect(db_path)
#             c = conn.cursor()
#             c.execute('''CREATE TABLE IF NOT EXISTS uniform_records
#                          (employee_id TEXT,
#                           bakery_id TEXT,
#                           date TEXT,
#                           uniform_info TEXT,
#                           compliance TEXT,
#                           image TEXT,
#                           PRIMARY KEY (employee_id, bakery_id, date))''')
            
#             employee_id = pid[:-1]
#             date = today
#             uniform_info = uniform_str
#             compliance = "No" if uniform_info == "none" or "No Uniform" in uniform_info else "Yes"
#             image = annotated_path
            
#             c.execute("INSERT OR REPLACE INTO uniform_records VALUES (?, ?, ?, ?, ?, ?)",
#                       (employee_id, bakery_id, date, uniform_info, compliance, image))
#             conn.commit()
#             conn.close()
#             print(f"    Inserted into DB: {employee_id}, {bakery_id}, {date}, Compliance: {compliance}")
            
#             person_found = True
#             break
    
#     if not person_found:
#         print(f"No matching person box found for {pid}")
#         # Still write an entry with no uniforms
#         result_line = f"{pid}_{today} : none"
#         final_results_file = os.path.join(final_results_dir, f"{today}.txt")
#         with open(final_results_file, "a") as f:
#             f.write(result_line + "\n")
        
#         # Insert to DB with none
#         db_path = os.path.join("Bakeries", "Uniform_information.db")
#         conn = sqlite3.connect(db_path)
#         c = conn.cursor()
#         c.execute('''CREATE TABLE IF NOT EXISTS uniform_records
#                      (employee_id TEXT,
#                       bakery_id TEXT,
#                       date TEXT,
#                       uniform_info TEXT,
#                       compliance TEXT,
#                       image TEXT,
#                       PRIMARY KEY (employee_id, bakery_id, date))''')
        
#         employee_id = pid[:-1]
#         date = today
#         uniform_info = "none"
#         compliance = "No"
#         image = ""  # No image if no person found, or set to None
        
#         c.execute("INSERT OR REPLACE INTO uniform_records VALUES (?, ?, ?, ?, ?, ?)",
#                   (employee_id, bakery_id, date, uniform_info, compliance, image))
#         conn.commit()
#         conn.close()




# _______________________________________________________________________________________________________________________________



# detect_person.py - Updated to use DINOv2 verification

import os
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import sqlite3
from uniform_ver import verify_uniform_single  # NEW: Import verification function

# === CONFIGURATION ===
MODEL_PATH = "Helping Files/yolo11s.pt"
UNIFORM_MODEL_PATH = "Helping Files/best.pt"
CLASSES_PATH = "Helping Files/classes.txt"
CONFIDENCE_THRESHOLD = 0.40

# Define uniform classes
ACTUAL_UNIFORM_CLASSES = {
    "Manager Uniform",
    "Salesman Uniform 1", 
    "Salesman Uniform 2",
    "Sweeper Uniform"
}

EXCLUSION_CLASSES = {
    "No Uniform"
}
# ======================

def load_classes(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def get_class_colors(class_names):
    """Generate consistent colors for each class"""
    np.random.seed(42)
    colors = {}
    for i, class_name in enumerate(class_names):
        colors[class_name] = tuple(map(int, np.random.randint(0, 255, 3)))
    return colors

def parse_loc_file(path, target_pid):
    """Parse face location from file"""
    face_bbox = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pid_date, coords = line.split(":", 1)
            current_pid = pid_date.strip().rsplit("_", 1)[0]
            if current_pid == target_pid:
                parts = coords.replace(" ", "").split(",")
                x = int(parts[0].split("=")[1])
                y = int(parts[1].split("=")[1])
                w = int(parts[2].split("=")[1])
                h = int(parts[3].split("=")[1])
                face_bbox = (x, y, w, h)
                break
    return face_bbox

def parse_uniform_bbox_file(path, target_pid):
    """Parse uniform bounding boxes from file"""
    uniforms = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pid_date, bbox_info = line.split(":", 1)
            current_pid = pid_date.strip().rsplit("_", 1)[0]
            if current_pid == target_pid:
                if bbox_info.strip() != "none":
                    uniform_parts = bbox_info.split(" | ")
                    for part in uniform_parts:
                        part = part.strip()
                        if not part:
                            continue
                        
                        class_conf, coords = part.split(":", 1)
                        class_name = class_conf.split("(")[0]
                        conf = float(class_conf.split("(")[1].split(")")[0])
                        
                        coord_parts = coords.split(",")
                        x = int(coord_parts[0].split("=")[1])
                        y = int(coord_parts[1].split("=")[1])
                        w = int(coord_parts[2].split("=")[1])
                        h = int(coord_parts[3].split("=")[1])
                        
                        uniforms.append((class_name, conf, x, y, w, h))
                break
    return uniforms

def boxes_overlap(box1, box2):
    """Check if two bounding boxes overlap"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    box1_x2, box1_y2 = x1 + w1, y1 + h1
    box2_x2, box2_y2 = x2 + w2, y2 + h2
    
    return not (box1_x2 <= x2 or box2_x2 <= x1 or box1_y2 <= y2 or box2_y2 <= y1)

def calculate_overlap_area(box1, box2):
    """Calculate the overlap area between two boxes"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    box1_x2, box1_y2 = x1 + w1, y1 + h1
    box2_x2, box2_y2 = x2 + w2, y2 + h2
    
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0
    
    return (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

def filter_uniform_results(person_uniforms):
    """Filter uniform results based on business logic"""
    has_actual_uniform = any(class_name in ACTUAL_UNIFORM_CLASSES for class_name in person_uniforms.keys())
    
    if has_actual_uniform:
        filtered_uniforms = {
            class_name: conf for class_name, conf in person_uniforms.items() 
            if class_name not in EXCLUSION_CLASSES
        }
    else:
        filtered_uniforms = person_uniforms.copy()
    
    uniform_classes = {k: v for k, v in filtered_uniforms.items() if k in ACTUAL_UNIFORM_CLASSES}
    if len(uniform_classes) > 1:
        max_conf_uniform = max(uniform_classes.items(), key=lambda x: x[1])
        filtered_uniforms = {
            k: v for k, v in filtered_uniforms.items() 
            if k not in ACTUAL_UNIFORM_CLASSES or k == max_conf_uniform[0]
        }
    
    return filtered_uniforms

def person_detection_single(pid):
    """Process person detection for a single person"""
    bakery_id = pid[-1]

    # Set dynamic directories
    base_dir = os.path.join("Bakeries", bakery_id, "Captured")
    frames_dir = os.path.join(base_dir, "frames")
    loc_info_dir = os.path.join(base_dir, "loc_info")
    uniform_bbox_dir = os.path.join(base_dir, "uniform_bbox")
    final_results_dir = os.path.join(base_dir, "final_results")
    person_frames_dir = os.path.join(base_dir, "person_frame")
    annotated_frames_dir = os.path.join(base_dir, "annotated_frames")
    
    os.makedirs(final_results_dir, exist_ok=True)
    os.makedirs(person_frames_dir, exist_ok=True)
    os.makedirs(annotated_frames_dir, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    loc_file = os.path.join(loc_info_dir, f"{today}.txt")
    uniform_bbox_file = os.path.join(uniform_bbox_dir, f"{today}.txt")
    
    # Get face bbox
    face_bbox = parse_loc_file(loc_file, pid)
    if not face_bbox:
        print(f"[detect_person] No face bbox found for {pid}")
        return

    fx, fy, fw, fh = face_bbox
    
    # Get uniform detections
    uniform_detections = parse_uniform_bbox_file(uniform_bbox_file, pid)
    
    model = YOLO(MODEL_PATH)
    uniform_model = YOLO(UNIFORM_MODEL_PATH)
    class_names = load_classes(CLASSES_PATH)
    class_colors = get_class_colors(class_names)
    
    # STEP 1: Create annotated frame
    print(f"[detect_person] Step 1: Creating annotated frame for {pid}...")
    
    frame_path = os.path.join(frames_dir, f"{pid}_{today}.jpg")
    img = cv2.imread(frame_path)
    if img is None:
        print(f"[detect_person] Could not load image: {frame_path}")
        return

    annotated_img = img.copy()

    # Draw face box
    cv2.rectangle(annotated_img, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
    cv2.putText(annotated_img, f"Face: {pid}", (fx, fy - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw uniform detections
    uniform_result = uniform_model(img, conf=CONFIDENCE_THRESHOLD)[0]

    if getattr(uniform_result, "obb", None) and uniform_result.obb.data is not None:
        obb_array = uniform_result.obb.data.cpu().numpy()
        for row in obb_array:
            cid, conf = int(row[6]), float(row[5])
            if conf >= CONFIDENCE_THRESHOLD and 0 <= cid < len(class_names):
                x_center, y_center, width, height, angle = row[0], row[1], row[2], row[3], row[4]
                
                rect = ((x_center, y_center), (width, height), np.degrees(angle))
                box = cv2.boxPoints(rect)
                box = box.astype(np.int32)

                class_name = class_names[cid]
                color = class_colors[class_name]
                cv2.drawContours(annotated_img, [box], 0, color, 2)
                cv2.putText(annotated_img, f"{class_name}({conf:.2f})",
                           (int(x_center), int(y_center)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    elif getattr(uniform_result, "boxes", None) and getattr(uniform_result.boxes, "cls", None) is not None:
        boxes = uniform_result.boxes.xyxy.cpu().numpy()
        cls_idxs = uniform_result.boxes.cls.cpu().numpy().astype(int)
        confs = uniform_result.boxes.conf.cpu().numpy()
        
        for (x1, y1, x2, y2), cid, conf in zip(boxes, cls_idxs, confs):
            if conf >= CONFIDENCE_THRESHOLD and 0 <= cid < len(class_names):
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                class_name = class_names[cid]
                color = class_colors[class_name]
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_img, f"{class_name}({conf:.2f})", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw person boxes
    results = model(img, classes=[0])
    person_boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes.xyxy) > 0 else []
    
    for i, person_box in enumerate(person_boxes):
        x1, y1, x2, y2 = map(int, person_box)
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(annotated_img, f"Person_{i+1}", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Save annotated frame
    annotated_path = os.path.join(annotated_frames_dir, f"{pid}_{today}_annotated.jpg")
    cv2.imwrite(annotated_path, annotated_img)
    print(f"[detect_person] Saved annotated frame: {annotated_path}")

    # STEP 2: Match face to person and save crop
    print(f"[detect_person] Step 2: Matching face to person for {pid}...")
    
    results = model(img, classes=[0])
    boxes = results[0].boxes.xyxy
    classes = results[0].boxes.cls

    cx = fx + fw / 2
    cy = fy + fh / 2

    person_found = False
    for (x1, y1, x2, y2), cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            # Crop and save person
            person_crop = img[y1:y2, x1:x2]
            person_frame_path = os.path.join(person_frames_dir, f"{pid}_{today}.jpg")
            cv2.imwrite(person_frame_path, person_crop)
            print(f"[detect_person] Saved person crop: {person_frame_path}")
            
            # ===== NEW: STEP 3 - Run DINOv2 Uniform Verification =====
            print(f"[detect_person] Step 3: Running DINOv2 uniform verification for {pid}...")
            
            verification_result = verify_uniform_single(pid, bakery_id, today)
            
            dinov2_class = verification_result["final_class"]
            best_match = verification_result["best_match"]
            similarity = verification_result["similarity"]
            
            print(f"[detect_person] DINOv2 Result: {dinov2_class} (match: {best_match}, sim: {similarity:.4f})")
            
            # ===== Determine Final Compliance Based on DINOv2 =====
            if dinov2_class == "Uniform":
                compliance = "Yes"
                uniform_info = "Uniform"  # Simple for frontend
                detailed_info = f"DINOv2: {best_match} (similarity: {similarity:.4f})"
            else:
                compliance = "No"
                uniform_info = "No uniform"  # Simple for frontend
                detailed_info = f"DINOv2: No uniform detected (similarity: {similarity:.4f})"
            
            # Save result to final_results.txt
            result_line = f"{pid}_{today} : {detailed_info}"  # For file logging
            final_results_file = os.path.join(final_results_dir, f"{today}.txt")
            with open(final_results_file, "a") as f:
                f.write(result_line + "\n")
            
            print(f"[detect_person] Final result for {pid}: {uniform_info}, Compliance: {compliance}")
            
            # ===== Save to Database =====
            db_path = os.path.join("Bakeries", "Uniform_information.db")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS uniform_records
                         (employee_id TEXT,
                          bakery_id TEXT,
                          date TEXT,
                          uniform_info TEXT,
                          compliance TEXT,
                          image TEXT,
                          PRIMARY KEY (employee_id, bakery_id, date))''')
            
            employee_id = pid[:-1]
            
            c.execute("INSERT OR REPLACE INTO uniform_records VALUES (?, ?, ?, ?, ?, ?)",
                      (employee_id, bakery_id, today, uniform_info, compliance, annotated_path))
            conn.commit()
            conn.close()
            print(f"[detect_person] Inserted into DB: {employee_id}, Compliance: {compliance}")
            
            person_found = True
            break
    
    if not person_found:
        print(f"[detect_person] No matching person box found for {pid}")
        # Write default entry
        result_line = f"{pid}_{today} : none"
        final_results_file = os.path.join(final_results_dir, f"{today}.txt")
        with open(final_results_file, "a") as f:
            f.write(result_line + "\n")
        
        # Insert to DB
        db_path = os.path.join("Bakeries", "Uniform_information.db")
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS uniform_records
                     (employee_id TEXT,
                      bakery_id TEXT,
                      date TEXT,
                      uniform_info TEXT,
                      compliance TEXT,
                      image TEXT,
                      PRIMARY KEY (employee_id, bakery_id, date))''')
        
        employee_id = pid[:-1]
        c.execute("INSERT OR REPLACE INTO uniform_records VALUES (?, ?, ?, ?, ?, ?)",
                  (employee_id, bakery_id, today, "none", "No", ""))
        conn.commit()
        conn.close()