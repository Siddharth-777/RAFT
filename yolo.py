import cv2
from ultralytics import YOLO
import numpy as np

def detect_ambulance(video_path):
    # Select device
    print(f"Using device: {'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'}")

    # Load YOLOv8 model
    model = YOLO("yolov8s.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Inference with YOLOv8
        results = model(frame)

        # Process detections
        for result in results:
            boxes = result.boxes
            print(f"Number of detections: {len(boxes)}")  # Debug: total detections
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = model.names[cls]

                # Detect only "truck" with confidence >= 0.75
                if label == "truck" and conf >= 0.75:
                    print(f"Detected: {label}, Confidence: {conf:.2f}")  # Debug: truck detection

                    # Draw rectangle
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    # Display label and confidence
                    conf_text = f"{label} {conf:.2f}"
                    cv2.putText(frame, conf_text, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow("Truck Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_all_vehicles(video_path):
    # Select device
    print(f"Using device: {'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'}")

    # Load YOLOv8 model
    model = YOLO("yolov8s.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Inference with YOLOv8
        results = model(frame)

        # Process detections
        for result in results:
            boxes = result.boxes
            print(f"Number of detections: {len(boxes)}")  # Debug: total detections
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = model.names[cls]

                # Detect all vehicle classes with confidence >= 0.75
                if label in ["car", "truck", "bus", "motorcycle","bike"] and conf >= 0.75:
                    print(f"Detected: {label}, Confidence: {conf:.2f}")  # Debug: vehicle detection

                    # Draw rectangle
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    # Display label and confidence
                    conf_text = f"{label} {conf:.2f}"
                    cv2.putText(frame, conf_text, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow("Vehicle Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()