import torch
import cv2

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # You can choose 'yolov5m', 'yolov5l', or 'yolov5x' for larger models

cap = cv2.VideoCapture(0)  

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read() 
    if not ret:
        break

    results = model(frame)

    frame = results.render()[0] 

    cv2.imshow('YOLOv5 Real-Time Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
