import torch
from PIL import Image
import cv2
import numpy as np

# Load a pretrained YOLOv5 model from the official repository
# This will download the YOLOv5 model automatically if not already present
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 'yolov5s' is the small version of the model

def detect_objects(image_path):
    img = Image.open(image_path)

    results = model(img)

    results.show()  

    print(results.pandas().xywh)  

    results.save()

image_path = 'D:\general\cheverlot1.jpg'
image_path2 = "D:\general\IMG_0147.JPG"
image_path3 = "D:\general\کارت ملی.jpg"
detect_objects(image_path2)
