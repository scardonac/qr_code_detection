from ultralytics import YOLO

# Cargar el modelo YOLOv8 preentrenado
model = YOLO('yolov8n.pt')  

# Entrenar el modelo
results = model.train(
    data=r'D:\qr_code_detection\model\qr_detection_dataset\data.yaml', 
    epochs=30,         
    imgsz=640,          
    batch=-1,          
    name='qr_code_detection', 
    augment=False,
    patience=5
    )