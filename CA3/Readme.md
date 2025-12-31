# Advanced Computer Vision Pipeline (Classification • Detection • Segmentation)

## Overview

This project demonstrates a **full-stack computer vision pipeline** built using **PyTorch**, **Torchvision**, and **timm**, covering:

* **Image Classification** using EfficientNet-B4
* **Object Detection** using Faster R-CNN ResNet50-FPN-v2
* **Semantic Segmentation** using DeepLabV3+ with a ResNet50 backbone
* A custom **12-step real-world preprocessing pipeline** for noisy images
* Visualization and interpretability for each task
* Evaluation on multiple sample images

The project showcases applied deep-learning engineering skills suitable for **industry-grade ML**, **AI research**, and **computer vision production pipelines**.

---

## Key Technical Skills Demonstrated

### 🔹 **1. Advanced Image Preprocessing Pipeline (12-step)**

A robust preprocessing system designed to handle real images from uncontrolled environments, incorporating:

* OpenCV/PIL interoperability
* Denoising (Gaussian Blur)
* Sharpening (+ Unsharp Masking)
* Local contrast enhancement (CLAHE)
* Smart geometric transforms: padding, resizing, random rotation, cropping
* Photometric transforms: color jitter, grayscale
* Tensor conversion and ImageNet normalization

This pipeline shows capability in preparing data for **high-performance inference and training**.

---

### 🔹 **2. Image Classification with EfficientNet-B4 (timm)**

Implements a full classifier using:

* `timm.create_model("efficientnet_b4", pretrained=True)`
* ImageNet-1K class mapping
* Preprocessing, normalization, and softmax inference
* Top-K class predictions
* Visualization of the raw vs. preprocessed image

Demonstrates understanding of **state-of-the-art CNN architectures**, pretrained transfer learning, and inference optimization.

---

### 🔹 **3. Object Detection with Faster R-CNN ResNet50-FPN-v2**

Complete object detection workflow using Torchvision’s modern detection model:

* Pretrained COCO weights
* CPU-safe inference mode
* Bounding box visualization
* Confidence scoring & post-processing
* Evaluation on multiple images

This highlights experience with **two-stage detection pipelines**, region proposals, and feature pyramids.

---

### 🔹 **4. Semantic Segmentation with DeepLabV3+ (ResNet50)**

Implementation includes:

* COCO + VOC label mapping
* High-quality preprocessing
* Pixel-level argmax segmentation
* Color-mapped masks using VOC palette
* Overlay visualization
* Evaluation on challenging images

This demonstrates proficiency in **dense prediction models** and pixel-wise deep learning.

---

## 🛠 Technologies Used

* **Python**
* **PyTorch**
* **Torchvision**
* **timm**
* **OpenCV**
* **Pillow (PIL)**
* **Matplotlib**
* **NumPy**
* **Ultralytics (YOLO support extension)**

---

## 🏅 Author

**Mehdy Mokhtari**
Machine Learning & NLP Engineer
Computer Vision • Deep Learning • AI Systems Engineering

