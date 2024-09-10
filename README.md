# QR Detection and Information Extraction

This project implements a **QR code detection** system using the **YOLOv8n** model along with a simple frontend built in **Streamlit**. The system is capable of detecting QR codes in images, extracting their content, and drawing bounding boxes around the detected QR codes.

## Table of Contents

2. [Model Used](#model-used)
   - [Why YOLOv8n Pre-trained Model?](#why-yolov8n-pre-trained-model)
3. [Data Used](#data-used)
4. [Data Augmentation](#data-augmentation)
5. [Model Training](#model-training)
   - [Training Results](#training-results)
   - [Ultralytics Generated Images](#ultralytics-generated-images)
6. [Running the Project with Docker](#running-the-project-with-docker)
7. [Using the Application in Streamlit](#using-the-application-in-streamlit)
8. [Requirements](#requirements)

## Model Used

The model used is **YOLOv8n** (You Only Look Once Nano), a state-of-the-art object detection model optimized for lightweight, real-time applications. 

### Why YOLOv8n Pre-trained Model?

The **YOLOv8n.pt** pre-trained model was chosen for the following reasons:

- **Lightweight**: YOLOv8n is the "nano" version of the YOLO family, optimized for speed and efficiency with smaller model sizes, making it ideal for real-time applications on devices with limited computational resources.
- **Transfer Learning**: By using a pre-trained model, we leverage the general object detection capabilities learned from a large dataset (such as COCO), which helps the model generalize better and converge faster when fine-tuned on a smaller, domain-specific dataset like QR codes.
- **Real-Time Performance**: YOLOv8n is specifically designed to handle real-time object detection tasks, ensuring minimal latency during inference, which is critical for QR code scanning applications.

## Data Used

The dataset used for this project was sourced from [Roboflow's QR Code Detection Dataset](https://universe.roboflow.com/qr-lmsul/qr-code-detection-jz2e3/dataset/2), which includes **1547 images** annotated in YOLOv8 format. This dataset was selected due to its diversity of QR code images, enabling the model to generalize better to different real-world scenarios.

## Data Augmentation

To further improve the model's ability to generalize, several **data augmentation techniques** were applied to the dataset. Data augmentation helps simulate different real-world conditions and improves model robustness. The following transformations were applied to 20% of the images:

- Horizontal and Vertical Flip: Helps the model handle QR codes that may appear upside down or flipped.
- Random Brightness Contrast: Simulates different lighting conditions, ensuring that the model performs well in various lighting environments.
- Gaussian Blur: Reduces overfitting by slightly blurring some images, ensuring the model doesn't rely on sharpness alone.
- Rotation: Randomly rotates the images by up to 10 degrees to handle minor rotations in QR code images.

These transformations allow the model to generalize across different variations in the appearance of QR codes, making it robust to real-world changes in QR code orientation, lighting, and noise.

## Model Training
The model was trained using the [Ultralytics YOLOv8 framework](https://www.ultralytics.com/es). The training process involved a dataset of annotated QR code images. The following hyperparameters were used:

- Image Size: 640x640 pixels
- Epochs: 30
- Optimizer: auto
- Patiente: 5

  ### Training Results

