from flask import Flask, render_template, Response, request
import cv2
from ultralytics import YOLO
import requests
import logging
app = Flask(__name__)
# Configure logging
logging.basicConfig(level=logging.DEBUG)
# Load the YOLO model
model = YOLO("best_v8_25.pt")  # Path to the pre-trained YOLOv5 nano model
# Initialize video capture object
cap = None
# Slack webhook URL
SLACK_WEBHOOK_URL = 'https://hooks.slack.com/services/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'
def send_slack_notification(message):
    payload = {'text': message}
    response = requests.post(SLACK_WEBHOOK_URL, json=payload)
    if response.status_code != 200:
        logging.error(f"Failed to send Slack notification: {response.status_code}, {response.text}")
    else:
        logging.info("Slack notification sent successfully")
def generate_frames():
    global cap
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        else:
            # Perform inference on the frame
            results = model(frame)
            helmet_detected = False
            # Draw the bounding boxes and labels on the frame
            for result in results:
                for detection in result.boxes.data.tolist():
                    x1, y1, x2, y2, conf, class_id = map(int, detection[:6])
                    label = f"{model.names[class_id]} {conf:.2f}"
                    color = (0, 255, 0)  # Bounding box color (green)
                    if model.names[class_id] == "helmet":
                        helmet_detected = True
                    # Draw the bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    # Draw the label
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if not helmet_detected:
                message = "Warning: Person detected without a helmet!"
                send_slack_notification(message)
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)  # Initialize video capture object
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/toggle_video', methods=['POST'])
def toggle_video():
    global cap
    if cap is not None and cap.isOpened():
        cap.release()  # Release the video capture object
        cap = None
    else:
        cap = cv2.VideoCapture(0)  # Initialize video capture object
    return '', 204
if __name__ == "__main__":
    app.run(debug=True)
