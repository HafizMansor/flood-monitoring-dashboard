import streamlit as st
import cv2
import numpy as np
import time
import platform
import os
import torch
from ultralytics import YOLO
import supervision as sv

# === SETTINGS ===
RTSP_URL = "rtsp://Mono_I:Hafiz1144@192.168.136.238:554/stream1"
MODEL_PATH = "runs/detect/train22/weights/best.pt"
WATER_LEVEL_THRESHOLD = 80
FPS_UPDATE_INTERVAL = 5  # Update chart every 5 seconds

# === Device Selection ===
device = 0 if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {'GPU (CUDA)' if device == 0 else 'CPU'}")

# === Load YOLO Model ===
model = YOLO(MODEL_PATH)
model.to(device)

# === Streamlit Page Config ===
st.set_page_config(page_title="Flood Monitoring Dashboard", layout="wide")
st.title("Flood Monitoring Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.header("Live CCTV Feed with Detection")
    frame_placeholder = st.empty()

with col2:
    st.header("Current Water Level")
    water_level_placeholder = st.empty()

st.header("Water Level Trend Over Time")
chart_placeholder = st.line_chart([])

# === Utility Functions ===
def beep():
    if platform.system() == 'Windows':
        import winsound
        winsound.Beep(1000, 500)
    else:
        os.system('beep -f 1000 -l 500')

def get_water_level():
    # Replace with actual sensor input if available
    return np.random.randint(30, 100)

# === Supervision Annotators ===
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

def detect_objects_with_supervision(frame):
    resized = cv2.resize(frame, (416, 416))
    result = model.predict(resized, conf=0.25, device=device, stream=False, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)
    annotated = box_annotator.annotate(scene=resized.copy(), detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections)
    return annotated, detections

# === Monitoring Logic ===
if st.button('Start Monitoring'):
    levels = []
    last_alert_sent = False
    last_fps_update = time.time()

    cap = cv2.VideoCapture(RTSP_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        st.error("Failed to connect to RTSP stream.")
    else:
        try:
            for _ in range(60):  # Adjust duration as needed
                ret, frame = cap.read()
                if not ret or frame is None:
                    st.error("Failed to read frame from stream.")
                    break

                annotated_frame, detections = detect_objects_with_supervision(frame)
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB")

                # Water Level Logic
                water_level = get_water_level()
                levels.append(water_level)

                if water_level < 50:
                    water_level_placeholder.success(f"Water Level: {water_level}% (Safe)")
                elif water_level < WATER_LEVEL_THRESHOLD:
                    water_level_placeholder.warning(f"Water Level: {water_level}% (Warning)")
                else:
                    water_level_placeholder.error(f"Water Level: {water_level}% (DANGER!)")
                    if not last_alert_sent:
                        beep()
                        last_alert_sent = True

                # Update the water level trend chart at intervals
                if time.time() - last_fps_update >= FPS_UPDATE_INTERVAL:
                    chart_placeholder.line_chart(levels)
                    last_fps_update = time.time()

                time.sleep(1)

        except KeyboardInterrupt:
            st.warning("Monitoring stopped by user.")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("[INFO] Resources released.")
