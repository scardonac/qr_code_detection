from ultralytics import YOLO

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  

# Training the model
results = model.train(
    data=r'D:\qr_code_detection\models\qr_detection_dataset\data.yaml', 
    epochs=30,         
    imgsz=640,                  
    name='qr_code_detection', 
    augment=False,
    patience=5
    )
