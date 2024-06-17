import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
from ultralytics import YOLO
import numpy as np
from twilio.rest import Client

# Get server credentials from environment variables
account_sid = st.secrets["general"]["TWILIO_ACCOUNT_SID"]
auth_token = st.secrets["general"]["TWILIO_AUTH_TOKEN"]
client = Client(account_sid, auth_token)
token = client.tokens.create()

# Set up the Streamlit app
st.set_page_config(page_title="Safety Helmet Detection", page_icon=":construction_worker:", layout="wide")
st.title(":construction_worker: Safety Helmet Detection")
st.markdown("""
This application detects safety helmets in real-time, using the YOLOv8 model. Adjust the settings in the sidebar to customize the detection.
""")

# Sidebar for additional settings and information
st.sidebar.header("⚙️ Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.7, help="Adjust the confidence threshold for detections.")
draw_boxes = st.sidebar.checkbox("Draw Bounding Boxes", True, help="Toggle to enable/disable drawing bounding boxes.")
draw_circles = st.sidebar.checkbox("Draw Bounding Circles", False, help="Toggle to enable/disable drawing bounding circles.")

# Color pickers for bounding box colors
helmet_color = st.sidebar.color_picker("Helmet Color", "#00FF00", help="Choose color for helmet bounding boxes.")
no_helmet_color = st.sidebar.color_picker("No Helmet Color", "#FF0000", help="Choose color for no helmet bounding boxes.")

# Function to convert hex to BGR
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

# Convert hex colors to BGR for OpenCV
helmet_color_bgr = hex_to_bgr(helmet_color)
no_helmet_color_bgr = hex_to_bgr(no_helmet_color)

# Load the YOLO model
model = YOLO("v8_n_50e_best.pt")  # Path to the pre-trained YOLOv5 model

# Initialize session state for counts
if 'helmet_count' not in st.session_state:
    st.session_state.helmet_count = 0
if 'no_helmet_count' not in st.session_state:
    st.session_state.no_helmet_count = 0

# Function to process video frames
def video_frame_callback(frame):
    if frame is None:
        return None

    img = frame.to_ndarray(format="bgr24")

    # Resize the frame
    img = cv2.resize(img, (640, 480))

    # Perform inference on the resized frame
    results = model(img, conf=confidence_threshold)

    # Initialize counts
    helmet_count = 0
    no_helmet_count = 0

    # Draw bounding boxes or circles and labels on the resized frame
    for detection in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, class_id = map(int, detection[:6])

        if model.names[class_id] == "helmet":
            color = helmet_color_bgr  # Use the selected helmet color
            helmet_count += 1
        else:
            color = no_helmet_color_bgr  # Use the selected no helmet color
            no_helmet_count += 1

        # Draw the bounding box or circle
        if draw_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        elif draw_circles:
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            radius = (x2 - x1) // 2
            cv2.circle(img, center, radius, color, 2)

    # Display the counts on the frame
    cv2.putText(img, f"Helmet Count: {helmet_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, helmet_color_bgr, 2)
    cv2.putText(img, f"No Helmet Count: {no_helmet_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, no_helmet_color_bgr, 2)

    # Update session state counts
    st.session_state.helmet_count = helmet_count
    st.session_state.no_helmet_count = no_helmet_count

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# WebRTC streaming
ctx = webrtc_streamer(
    key="example",
    video_frame_callback=video_frame_callback,
    rtc_configuration = {
    "iceServers": token.ice_servers
                        },
    async_processing=True,
    media_stream_constraints={"video": True, "audio": False},  # Disable audio
)

# Information and instructions
st.sidebar.markdown("### ℹ️ Instructions")
st.sidebar.markdown("""
1. Adjust the confidence threshold to filter detections.
2. Toggle drawing bounding boxes or circles on/off.
3. Choose colors for the bounding boxes.
4. View the video stream with detection annotations.
""")

# Advertisement for developers' LinkedIn profiles
st.sidebar.markdown("### 🌟 Connect with the Developers on LinkedIn")
st.sidebar.markdown("""
Want to connect with the developers behind this project? Reach out to them on LinkedIn!
- **Nick**: [Connect](https://www.linkedin.com/in/nicholas-sayre-58b64264/)
- **Karush**: [Connect](https://www.linkedin.com/in/karushpradhan/)
- **Sayaka**: [Connect](https://www.linkedin.com/in/sayaka-tajima/)
- **Iliyas**: [Connect](https://www.linkedin.com/in/iliyas-kurmangali-432273188/)
""")
