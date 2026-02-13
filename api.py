# api.py

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import sqlite3
import threading
import queue
import os
import cv2
import base64
import asyncio
from datetime import datetime
from typing import Optional, List, Set
from pydantic import BaseModel
from detect_uniform import uniform_detection_single
from detect_person import person_detection_single

# --- Imports for GPU/Threading Optimization ---
import numpy as np
import insightface
import time
from camera_rtsp import rehbar_rtsp_url, rehbar_rtsp_url2, get_roi_coordinates

app = FastAPI(title="Uniform Compliance API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for pipeline control
processing_queue = queue.Queue()
stop_recognition = threading.Event()
frame_queue = queue.Queue(maxsize=20)
recognition_thread = None
processing_thread = None
pipeline_status = {"running": False, "processed_count": 0, "current_person": None}

# Persistent captured IDs that reset only on date change
captured_ids_global = set()
last_reset_date = None

# WebSocket connections
active_connections: Set[WebSocket] = set()

# Pydantic models
class UniformRecord(BaseModel):
    employee_id: str
    bakery_id: str
    date: str
    uniform_info: str
    compliance: str
    image: str

class StatusResponse(BaseModel):
    running: bool
    processed_count: int
    current_person: Optional[str]

def get_db_connection():
    db_path = os.path.join("Bakeries", "Uniform_information.db")
    
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS uniform_records
                     (employee_id TEXT,
                      bakery_id TEXT,
                      date TEXT,
                      uniform_info TEXT,
                      compliance TEXT,
                      image TEXT,
                      PRIMARY KEY (employee_id, bakery_id, date))''')
    conn.commit()
    
    return conn


def should_reset_captured_ids():
    """Check if captured_ids should be reset (new day)"""
    global last_reset_date
    today = datetime.now().strftime("%Y-%m-%d")
    
    if last_reset_date is None or last_reset_date != today:
        return True
    return False


def reset_captured_ids_if_needed():
    """Reset captured_ids if date has changed"""
    global captured_ids_global, last_reset_date
    
    if should_reset_captured_ids():
        today = datetime.now().strftime("%Y-%m-%d")
        print(f"[Date Change] Resetting captured_ids. Old date: {last_reset_date}, New date: {today}")
        captured_ids_global.clear()
        last_reset_date = today


def process_person_pipeline():
    """Background thread for processing persons"""
    global pipeline_status
    processed_persons = set()
    
    while not stop_recognition.is_set():
        try:
            pid = processing_queue.get(timeout=1.0)
            
            if pid in processed_persons:
                continue
            
            pipeline_status["current_person"] = pid
            print(f"Processing pipeline for {pid}...")
            
            uniform_detection_single(pid)
            person_detection_single(pid)
            
            processed_persons.add(pid)
            pipeline_status["processed_count"] += 1
            pipeline_status["current_person"] = None
            print(f"Completed processing for {pid}")
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in processing pipeline: {e}")


def recognize_with_streaming(processing_queue, stop_recognition, frame_queue):
    """
    Recognition function optimized for GPU with ROI filtering for faces only.
    Only processes faces detected within the ROI defined in camera_rtsp.py
    """
    
    global captured_ids_global
    
    GALLERY_PATH = "Helping Files/Testing_no_uniform.npz"
    THRESHOLD = 0.4
    USE_GPU = True
    RECOGNITION_INTERVAL = 15
    FRAME_SKIP_CAPTURE = 1

    # Get ROI coordinates from camera_rtsp.py
    roi_x1_abs, roi_y1_abs, roi_x2_abs, roi_y2_abs = get_roi_coordinates()

    def load_model():
        app = insightface.app.FaceAnalysis(name="buffalo_l")
        ctx_id = 0 if USE_GPU else -1 
        try:
            app.prepare(ctx_id=ctx_id)
        except Exception:
            print("Failed to use requested GPU. Falling back to CPU.")
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

    def is_face_in_roi(bbox, scale_factor=1.0):
        """
        Check if face bounding box center is within ROI.
        
        Args:
            bbox: Face bounding box [x1, y1, x2, y2]
            scale_factor: Scale factor to convert scaled coordinates to original
        
        Returns:
            True if face center is within ROI, False otherwise
        """
        x1, y1, x2, y2 = bbox
        
        # Convert to original frame coordinates if scaled
        x1_orig = x1 * scale_factor
        y1_orig = y1 * scale_factor
        x2_orig = x2 * scale_factor
        y2_orig = y2 * scale_factor
        
        # Calculate face center point
        face_center_x = (x1_orig + x2_orig) / 2
        face_center_y = (y1_orig + y2_orig) / 2
        
        # Check if center is within ROI
        in_roi = (roi_x1_abs <= face_center_x <= roi_x2_abs and 
                  roi_y1_abs <= face_center_y <= roi_y2_abs)
        
        return in_roi

    gallery = load_gallery(GALLERY_PATH)
    app = load_model()
    frame_counter = 0

    cap = cv2.VideoCapture(rehbar_rtsp_url2)
    if not cap.isOpened():
        print("Failed to open RTSP feed.")
        return

    prev_time = time.time()
    start_time = time.time()
    
    # Get frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Frame dimensions: {frame_width}x{frame_height}")
    print(f"Face Recognition ROI (absolute coordinates): ({roi_x1_abs},{roi_y1_abs}) to ({roi_x2_abs},{roi_y2_abs})")
    print(f"ROI Size: {roi_x2_abs - roi_x1_abs}x{roi_y2_abs - roi_y1_abs}")
    print(f"NOTE: Faces outside ROI will be completely ignored")
    
    while not stop_recognition.is_set():
        # Check if date changed and reset if needed
        reset_captured_ids_if_needed()
        
        for _ in range(FRAME_SKIP_CAPTURE):
            if not cap.grab():
                break

        ret, frame = cap.read()

        if not ret or frame is None:
            print("⚠ Failed to read frame. Skipping...")
            continue

        # Draw ROI on frame for streaming
        display_frame = frame.copy()
        
        # Draw ROI rectangle (yellow color for face recognition zone)
        cv2.rectangle(display_frame, 
                     (roi_x1_abs, roi_y1_abs), 
                     (roi_x2_abs, roi_y2_abs), 
                     (0, 255, 255),  # Yellow in BGR
                     3)
        
        # Add label
        cv2.putText(display_frame, "Face Recognition Zone", 
                   (roi_x1_abs + 10, roi_y1_abs + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8,
                   (0, 255, 255),
                   2)

        try:
            if not frame_queue.full():
                frame_queue.put_nowait(display_frame) 
        except:
            pass

        if frame_counter % RECOGNITION_INTERVAL == 0:
            
            orig = frame.copy()  # Original frame for saving
            
            SCALE_FACTOR = 0.5
            process_frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            
            # Detect all faces
            all_faces = app.get(process_frame)
            
            # Filter faces by ROI (scale factor is inverse since we scaled down the frame)
            inverse_scale = 1.0 / SCALE_FACTOR
            faces_in_roi = [face for face in all_faces 
                           if is_face_in_roi(face.bbox, inverse_scale)]
            
            # Only process faces that are in ROI
            if len(all_faces) > len(faces_in_roi):
                print(f"Detected {len(all_faces)} faces, {len(faces_in_roi)} in ROI (ignoring {len(all_faces) - len(faces_in_roi)} outside)")

            for face in faces_in_roi:
                # All faces here are guaranteed to be in ROI
                emb = face.normed_embedding
                pid, score = recognize(emb, gallery, THRESHOLD)
                
                if not pid or pid in captured_ids_global:
                    continue
                
                # Check if already in database for today
                today = datetime.now().strftime("%Y-%m-%d")
                employee_id = pid[:-1]
                bakery_id = pid[-1]
                
                try:
                    conn = get_db_connection()
                    cursor = conn.execute(
                        "SELECT COUNT(*) as count FROM uniform_records WHERE employee_id = ? AND bakery_id = ? AND date = ?",
                        (employee_id, bakery_id, today)
                    )
                    already_recorded = cursor.fetchone()["count"] > 0
                    conn.close()
                    
                    if already_recorded:
                        print(f"⚠ {pid} already recorded today ({today}). Skipping.")
                        captured_ids_global.add(pid)
                        continue
                except Exception as e:
                    print(f"Database check error for {pid}: {e}")
                    continue
                
                # Get face coordinates for drawing and saving
                inv_scale = 1.0 / SCALE_FACTOR
                x1, y1, x2, y2 = (face.bbox * inv_scale).astype(int)
                
                # Draw green box on display frame
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, "DETECTED", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                base_dir = os.path.join("Bakeries", bakery_id, "Captured")
                frames_dir = os.path.join(base_dir, "frames")
                loc_info_dir = os.path.join(base_dir, "loc_info")
                os.makedirs(frames_dir, exist_ok=True)
                os.makedirs(loc_info_dir, exist_ok=True)

                today = datetime.now().strftime("%Y-%m-%d")
                
                # Save FULL frame (OVERWRITE if exists)
                img_path = os.path.join(frames_dir, f"{pid}_{today}.jpg")
                cv2.imwrite(img_path, orig)

                # Update loc_info file - replace old entry with new coordinates
                w, h = x2 - x1, y2 - y1
                new_line = f"{pid}_{today} : x={x1},y={y1}, w={w}, h={h}"
                loc_file_path = os.path.join(loc_info_dir, f"{today}.txt")
                
                # Read existing lines
                existing_lines = []
                if os.path.exists(loc_file_path):
                    with open(loc_file_path, "r") as f:
                        existing_lines = f.readlines()
                
                # Remove old entry for this person if exists
                existing_lines = [line for line in existing_lines if not line.startswith(f"{pid}_{today}")]
                
                # Add new entry
                existing_lines.append(new_line + "\n")
                
                # Write back all lines
                with open(loc_file_path, "w") as f:
                    f.writelines(existing_lines)
                
                print(f"✓ Updated coordinates for {pid} in ROI: {img_path}")

                captured_ids_global.add(pid)
                print(f"✓ Captured {pid} in ROI: {img_path}")
                
                # Trigger pipeline
                processing_queue.put(pid)
                
                # Update display frame with recognized person info
                cv2.putText(display_frame, f"{pid} ({score:.2f})", (x1, y2+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw FPS on display frame
        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-6)
        prev_time = now
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        frame_counter += 1

    cap.release()
    stop_recognition.set()

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

# ----------------- Pipeline Control Endpoints -----------------

@app.post("/api/start")
async def start_pipeline():
    """Start the recognition pipeline"""
    global recognition_thread, processing_thread, pipeline_status
    
    if pipeline_status["running"]:
        raise HTTPException(status_code=400, detail="Pipeline already running")
    
    # Check and reset captured_ids if date changed
    reset_captured_ids_if_needed()
    
    # Reset
    stop_recognition.clear()
    pipeline_status = {"running": True, "processed_count": 0, "current_person": None}
    
    # Clear frame queue
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except:
            break
    
    # Start threads
    processing_thread = threading.Thread(target=process_person_pipeline, daemon=True)
    processing_thread.start()
    
    recognition_thread = threading.Thread(
        target=recognize_with_streaming,
        args=(processing_queue, stop_recognition, frame_queue),
        daemon=True
    )
    recognition_thread.start()
    
    return {"message": "Pipeline started successfully", "status": pipeline_status}

@app.post("/api/stop")
async def stop_pipeline():
    """Stop the recognition pipeline"""
    global pipeline_status
    
    if not pipeline_status["running"]:
        raise HTTPException(status_code=400, detail="Pipeline not running")
    
    stop_recognition.set()
    pipeline_status["running"] = False
    
    return {"message": "Pipeline stopped successfully", "status": pipeline_status}

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get current pipeline status"""
    return pipeline_status

# ----------------- WebSocket Streaming -----------------

@app.websocket("/ws/stream")
async def stream_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for live video streaming.
    Optimized for low latency by aggressively draining the queue.
    """
    await websocket.accept()
    active_connections.add(websocket)
    
    try:
        while True:
            # Anti-Lag Optimization: Drain the queue to get only the NEWEST frame
            latest_frame = None
            while not frame_queue.empty():
                try:
                    latest_frame = frame_queue.get_nowait()
                except queue.Empty:
                    break 

            if latest_frame is not None:
                
                # Streaming Frame Preparation
                
                # 1. Resize to a smaller streaming resolution for faster transfer
                stream_frame = cv2.resize(latest_frame, (800, 450), interpolation=cv2.INTER_LINEAR)
                
                # 2. Aggressive Encoding: Use low JPEG quality to reduce file size
                _, buffer = cv2.imencode('.jpg', stream_frame, [cv2.IMWRITE_JPEG_QUALITY, 60]) 
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                
                # Send to client
                await websocket.send_json({"frame": jpg_as_text})
            
            # Polling frequency: ~66 FPS for low latency
            await asyncio.sleep(0.015) 
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        active_connections.discard(websocket)

# ----------------- Data Endpoints -----------------

@app.get("/api/records", response_model=List[UniformRecord])
async def get_records(
    date: Optional[str] = Query(None, description="Filter by date (YYYY-MM-DD)"),
    bakery_id: Optional[str] = Query(None, description="Filter by bakery ID"),
    employee_id: Optional[str] = Query(None, description="Filter by employee ID"),
    compliance: Optional[str] = Query(None, description="Filter by compliance (Yes/No)")
):
    """Get uniform records with optional filters"""
    conn = get_db_connection()
    
    query = "SELECT * FROM uniform_records WHERE 1=1"
    params = []
    
    if date:
        query += " AND date = ?"
        params.append(date)
    if bakery_id:
        query += " AND bakery_id = ?"
        params.append(bakery_id)
    if employee_id:
        query += " AND employee_id = ?"
        params.append(employee_id)
    if compliance:
        query += " AND compliance = ?"
        params.append(compliance)
    
    query += " ORDER BY date DESC, employee_id"
    
    cursor = conn.execute(query, params)
    records = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return records

@app.get("/api/image/{bakery_id}/{image_name}")
async def get_image(bakery_id: str, image_name: str):
    """Get image file"""
    if image_name.startswith("Bakeries"):
        image_path = image_name
    else:
        image_path = os.path.join("Bakeries", bakery_id, "Captured", "annotated_frames", image_name)
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")
    
    return FileResponse(image_path)

@app.get("/api/summary")
async def get_summary():
    """Get compliance summary statistics"""
    conn = get_db_connection()
    
    total = conn.execute("SELECT COUNT(*) as count FROM uniform_records").fetchone()["count"]
    compliant = conn.execute("SELECT COUNT(*) as count FROM uniform_records WHERE compliance = 'Yes'").fetchone()["count"]
    non_compliant = conn.execute("SELECT COUNT(*) as count FROM uniform_records WHERE compliance = 'No'").fetchone()["count"]
    
    by_bakery = conn.execute("""
        SELECT bakery_id, 
               COUNT(*) as total,
               SUM(CASE WHEN compliance = 'Yes' THEN 1 ELSE 0 END) as compliant
        FROM uniform_records
        GROUP BY bakery_id
    """).fetchall()
    
    conn.close()
    
    return {
        "total_records": total,
        "compliant": compliant,
        "non_compliant": non_compliant,
        "compliance_rate": round((compliant / total * 100) if total > 0 else 0, 2),
        "by_bakery": [dict(row) for row in by_bakery]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)