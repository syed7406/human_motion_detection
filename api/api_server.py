import os
import sys
import threading
import time
import cv2
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from main import HumanMotionDetectionSystem

app = FastAPI(title="Human Motion Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

latest_frame_jpeg = None
latest_info = {}
camera_active = False
stop_event = threading.Event()


def detector_loop(source=0, interval=0.03):
    global latest_frame_jpeg, latest_info, camera_active
    system = HumanMotionDetectionSystem()
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open source {source}")
        camera_active = False
        latest_info = {
            "camera_active": False,
            "status": "camera_unavailable"
        }
        return

    camera_active = True

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        processed = system.process_frame(frame)

        # encode to jpeg
        _, jpeg = cv2.imencode('.jpg', processed)
        latest_frame_jpeg = jpeg.tobytes()

        latest_info = {
            "fps": system.fps,
            "frame_count": system.frame_count,
            "total_detected": system.tracker.next_object_id,
            "camera_active": True
        }

        time.sleep(interval)

    cap.release()
    camera_active = False


@app.on_event("startup")
def startup_event():
    t = threading.Thread(target=detector_loop, args=(0, 0.03), daemon=True)
    t.start()


@app.on_event("shutdown")
def shutdown_event():
    stop_event.set()


@app.get("/frame")
def get_frame():
    if latest_frame_jpeg is None:
        return Response(status_code=204)
    return Response(content=latest_frame_jpeg, media_type="image/jpeg")


@app.get("/status")
def status():
    base = {
        "fps": None,
        "frame_count": None,
        "total_detected": None,
        "camera_active": camera_active
    }
    if latest_info:
        base.update(latest_info)
    else:
        base["status"] = "warmup"
    return base


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
