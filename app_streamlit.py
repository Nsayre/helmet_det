from flask import Flask, render_template, Response, request
import cv2
from ultralytics import YOLO
app = Flask(__name__)
# Load the YOLO model
model = YOLO("yolo_v8_25.pt")  # Path to the pre-trained YOLOv5 nano model
# Initialize video capture object
cap = None
def generate_frames():
    global cap
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        else:
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
