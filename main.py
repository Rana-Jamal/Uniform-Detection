# main.py

import threading
import queue
import time
from recognize_realtime import recognize_face_threaded
from detect_uniform import uniform_detection_single
from detect_person import person_detection_single

# Global queue for communication between threads
processing_queue = queue.Queue()
stop_recognition = threading.Event()

def process_person_pipeline():
    """Thread function that processes uniform detection and person detection for each captured person"""
    processed_persons = set()
    
    while not stop_recognition.is_set():
        try:
            # Wait for a person to be captured (with timeout to check stop condition)
            pid = processing_queue.get(timeout=1.0)
            
            if pid in processed_persons:
                continue
                
            print(f"Processing pipeline for {pid}...")
            
            # Run uniform detection for this specific person
            uniform_detection_single(pid)
            
            # Run person detection for this specific person
            person_detection_single(pid)
            
            processed_persons.add(pid)
            print(f"Completed processing for {pid}")
            
        except queue.Empty:
            # Timeout occurred, continue to check stop condition
            continue
        except Exception as e:
            print(f"Error in processing pipeline: {e}")

def main():
    print("Starting multi-threaded face recognition and processing pipeline...")
    
    # Start the processing thread
    processing_thread = threading.Thread(target=process_person_pipeline, daemon=True)
    processing_thread.start()
    
    # Start face recognition (this will run until completion)
    recognition_thread = threading.Thread(
        target=recognize_face_threaded, 
        args=(processing_queue, stop_recognition), 
        daemon=True
    )
    recognition_thread.start()
    
    try:
        # Wait for recognition to complete
        recognition_thread.join()
        
        # Give processing thread some time to finish remaining items
        print("Recognition completed. Waiting for processing to finish...")
        time.sleep(2)
        
        # Signal processing thread to stop
        stop_recognition.set()
        processing_thread.join(timeout=5)
        
    except KeyboardInterrupt:
        print("Interrupted by user")
        stop_recognition.set()
    
    print("Pipeline completed!")

if __name__ == "__main__":
    main()