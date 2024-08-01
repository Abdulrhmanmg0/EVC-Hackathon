import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Load YOLO model
model = YOLO("C:\\Users\\abdul\\Downloads\\best (1).pt")

st.title("YOLO Object Detection from Webcam")


# Function to process frame with YOLO
def process_frame(frame):
    results = model(frame)
    try:
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
        confidences = results[0].boxes.conf.cpu().numpy()  # Get confidences
        class_ids = results[0].boxes.cls.cpu().numpy()  # Get class IDs
        labels = results.names  # Get class names

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = labels[int(class_id)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    except:
        return frame
    return frame

# Start webcam
cap = cv2.VideoCapture("C:\\Users\\abdul\\Pictures\\Camera Roll\\mask-no-gloves.mp4")

stframe = st.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, conf=0.5)
    frame = results[0].plot()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    frame = Image.fromarray(frame)

    stframe.image(frame, use_column_width=True)

cap.release()