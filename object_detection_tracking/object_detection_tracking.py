import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from matplotlib import pyplot as plt

# ---- Step 1: Load YOLOv5 Model ----
# This loads the YOLOv5s model (small version), which is fast and has good accuracy.
# Note: The torch.hub.load call downloads the model if not previously cached.
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

# ---- Step 2: Initialize Deep SORT Tracker ----
tracker = DeepSort(max_age=30)

# ---- Step 3: Open the Video Stream ----
# Change '0' to a file path if you want to use a video file instead of a webcam.
video_source = 0
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

print("Starting video stream. Press 'q' to quit.")

# ---- Step 4: Process Video Frames in Real Time ----
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB color space as required by the YOLOv5 model
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # ---- Object Detection ----
    results = model(img)
    detections = results.pred[0].cpu().numpy()  # detections: [x1, y1, x2, y2, conf, class]

    # Prepare data for the tracker
    track_inputs = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf < 0.4:  # Use a confidence threshold to filter weak detections
            continue
        # Deep SORT expects bounding boxes in [x, y, w, h] format
        bbox = [x1, y1, x2 - x1, y2 - y1]
        track_inputs.append((bbox, conf, str(int(cls))))

    # ---- Update Tracker ----
    tracks = tracker.update_tracks(track_inputs, frame=frame)

    # ---- Drawing Bounding Boxes and Labels ----
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        # Draw a bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Put a label with the tracking ID above the box
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ---- Display the Output in a Window ----
    cv2.imshow("YOLOv5 + Deep SORT Tracking", frame)
    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---- Step 5: Cleanup ----
cap.release()
cv2.destroyAllWindows()

# python object_detection_tracking.py
