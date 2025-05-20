from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import serial
import threading

app = Flask(__name__)

# Global variables
camera_active = False
cap = None
co2_ppm = 0 

# Initialize Serial Connection to Arduino
try:
    arduino = serial.Serial(port='COM5', baudrate=115200, timeout=1)  # Change port if needed (Windows: COM3)
except Exception as e:
    print(f"Error connecting to Arduino: {str(e)}")
    arduino = None

def read_co2_sensor():
    """ Continuously read CO₂ PPM from Arduino and update global variable """
    global co2_ppm
    while True:
        if arduino:
            try:
                line = arduino.readline().decode().strip()  # Read serial data
                if line.isdigit():  # Ensure valid numeric data
                    co2_ppm = int(line)
            except Exception as e:
                print(f"CO₂ Sensor Read Error: {str(e)}")
        time.sleep(1)  # Read every 1 second

# Start CO₂ Sensor Thread
co2_thread = threading.Thread(target=read_co2_sensor, daemon=True)
co2_thread.start()

def init_models():
    """ Initialize the YOLO model """
    model_yolo = YOLO("yolov8n.pt")
    print("YOLO model initialized successfully")
    return model_yolo

def init_camera():
    """ Try to initialize the camera """
    global cap
    for camera_index in [ 0,1, 2]:  # Adjust camera indices
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"Camera opened at index {camera_index}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 96)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 96)
            return True
    print("Error: Could not open any camera")
    return False

def release_camera():
    """ Release the camera resource """
    global cap
    if cap is not None and cap.isOpened():
        cap.release()
    cap = None

def generate_frames():
    """ Generate frames from the camera and apply object detection """
    global camera_active, cap
    model = init_models()
    SCALE_FACTOR = 6
    placeholder = np.zeros((96, 96, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Camera Off", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    while True:
        if not camera_active:
            # Show placeholder when camera is off
            enlarged_placeholder = cv2.resize(placeholder, 
                                        (96 * SCALE_FACTOR, 96 * SCALE_FACTOR),
                                        interpolation=cv2.INTER_NEAREST)
            ret, buffer = cv2.imencode('.jpg', enlarged_placeholder)
            if ret:
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
            continue
        
        if cap is None or not cap.isOpened():
            if not init_camera():
                time.sleep(0.1)
                continue
        
        ret, frame = cap.read()
        if not ret:
            break
        
        thermal_frame = cv2.applyColorMap(frame, cv2.COLORMAP_INFERNO)  # Apply color mapping
        results = model(frame)
        
        enlarged_frame = cv2.resize(thermal_frame, (96 * 5, 96 * 5))
        if results:
            boxes = results[0].boxes.xyxy.numpy()
            confidences = results[0].boxes.conf.numpy()
            class_idxs = results[0].boxes.cls.numpy()
            class_names = results[0].names

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                confidence = confidences[i]
                class_idx = class_idxs[i]
                class_name = class_names[class_idx]

                if class_name == 'person':
                    label = f"{class_name} ({confidence:.2f})"
                    cv2.rectangle(enlarged_frame, (x1*4, y1*4), (x2*4, y2*4), (0, 255, 255), 3)
                    cv2.putText(enlarged_frame, label, (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', enlarged_frame)
        if ret:
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    """ Render the index page """
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """ Stream video frames """
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/co2')
def get_co2():
    """ API to get latest CO₂ PPM value """
    return jsonify({"ppm": co2_ppm})

@app.route('/control', methods=['POST'])
def control():
    """ Start or stop the camera """
    global camera_active  # Declare as global
    action = request.form.get('action')
    print(f"Received action: {action}")
    
    if action == 'start':
        camera_active = True
    elif action == 'stop':
        camera_active = False
        release_camera()
    
    return '', 204

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
