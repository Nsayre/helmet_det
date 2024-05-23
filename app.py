import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
# Load the YOLO model
model = YOLO("best_v8_25.pt")  # Path to the pre-trained YOLOv5 nano model
# Set up the Streamlit app
st.title("Real-Time Object Detection with YOLOv8")
st.text("Press the button below to start the camera")
# Add a button to start/stop the camera
start_camera = st.button("Start/Stop Camera")
# Initialize variables
cap = None
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
# Function to run inference and display video frames
def run_camera():
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    while st.session_state.camera_active:
        success, frame = cap.read()
        if not success:
            break
        # Perform inference on the frame
        results = model(frame)
        # Draw the bounding boxes and labels on the frame
        for result in results:
            for detection in result.boxes.data.tolist():
                x1, y1, x2, y2, conf, class_id = map(int, detection[:6])
                label = f"{model.names[class_id]} {conf:.2f}"
                color = (0, 255, 0)  # Bounding box color (green)
                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # Draw the label
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # Convert the frame to RGB format for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels='RGB')
    cap.release()
# Handle the start/stop camera button
if start_camera:
    st.session_state.camera_active = not st.session_state.camera_active
    if st.session_state.camera_active:
        run_camera()
