# recognize_realtime.py

import os
import numpy as np
import cv2
import insightface
import time
from datetime import datetime
from camera_rtsp import office_rtsp_url, team_rtsp_url, rehbar_rtsp_url

# === CONFIGURATION ===
GALLERY_PATH = "Helping Files/Testing_no_uniform.npz"
THRESHOLD = 0.4
USE_GPU = True
DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080
TIME_LIMIT_SECONDS = 300 * 1000  # NEW: hard stop after * seconds
# =======================

def load_model():
    app = insightface.app.FaceAnalysis(name="buffalo_l")
    ctx_id = 0 if USE_GPU else -1
    try:
        app.prepare(ctx_id=ctx_id)
    except Exception:
        app.prepare(ctx_id=-1)
    return app

def load_gallery(path):
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}

def recognize(emb, gallery, threshold):
    best_score, best_id = -1.0, None
    for pid, emb_list in gallery.items():
        sims = emb_list @ emb
        idx = np.argmax(sims)
        if sims[idx] > best_score:
            best_score, best_id = float(sims[idx]), pid
    return (best_id, best_score) if best_score >= threshold else (None, best_score)

def recognize_face_threaded(processing_queue, stop_recognition):
    """Modified version of recognize_face that puts new captures in queue"""
    gallery = load_gallery(GALLERY_PATH)
    app = load_model()

    all_ids = set(gallery.keys())
    captured_ids = set()

    # cap = cv2.VideoCapture(rehbar_rtsp_url)
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(r"val.mp4")
    if not cap.isOpened():
        print("Failed to open RTSP feed.")
        return

    cv2.namedWindow("Real-time Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Real-time Recognition", DISPLAY_WIDTH, DISPLAY_HEIGHT)

    prev_time = time.time()
    start_time = time.time()  # NEW: start the timeout timer
    
    # Get total frame count for video files to detect when video ends
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else float('inf')
    current_frame = 0
    
    while not stop_recognition.is_set():
        cap.grab(); cap.grab()
        ret, frame = cap.read()
        current_frame += 2  # Since we grab twice
        
        # Check if video ended
        if not ret or current_frame >= total_frames:
            print("Video ended or failed to read frame. Exiting.")
            break

        orig = frame.copy()
        faces = app.get(frame)

        for face in faces:
            emb = face.normed_embedding
            pid, score = recognize(emb, gallery, THRESHOLD)
            if not pid or pid in captured_ids:
                continue

            # Parse EmployeeID and BakeryID from pid (assuming format like 026052R)
            employee_id = pid[:-1]
            bakery_id = pid[-1]

            # Set dynamic directories based on BakeryID
            base_dir = os.path.join("Bakeries", bakery_id, "Captured")
            frames_dir = os.path.join(base_dir, "frames")
            loc_info_dir = os.path.join(base_dir, "loc_info")
            os.makedirs(frames_dir, exist_ok=True)
            os.makedirs(loc_info_dir, exist_ok=True)

            today = datetime.now().strftime("%Y-%m-%d")

            # 1) save UNANNOTATED frame as "{pid}_{date}.jpg"
            img_path = os.path.join(frames_dir, f"{pid}_{today}.jpg")
            cv2.imwrite(img_path, orig)

            # 2) append one line to loc_info/YYYY-MM-DD.txt
            # bounding box
            x1, y1, x2, y2 = face.bbox.astype(int)
            w, h = x2 - x1, y2 - y1
            line = f"{pid}_{today} : x={x1},y={y1}, w={w}, h={h}"
            loc_file_path = os.path.join(loc_info_dir, f"{today}.txt")
            with open(loc_file_path, "a") as f:
                f.write(line + "\n")

            captured_ids.add(pid)
            print(f"Captured {pid}: {img_path}")
            
            # NEW: Put the person ID in queue for processing
            processing_queue.put(pid)

            # still draw on `frame` for display
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(frame, f"{pid} ({score:.2f})", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # overlay FPS
        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-6)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

        # show resized, annotated feed
        disp = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT),
                          interpolation=cv2.INTER_AREA)
        cv2.imshow("Real-time Recognition", disp)

        # once everyone's captured, exit
        if captured_ids == all_ids:
            print("All faces captured. Exiting.")
            break

        # NEW: stop after TIME_LIMIT_SECONDS regardless
        if (time.time() - start_time) >= TIME_LIMIT_SECONDS:
            print(f"Time limit of {TIME_LIMIT_SECONDS} seconds reached. Exiting.")
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    stop_recognition.set()  # Signal other threads to stop