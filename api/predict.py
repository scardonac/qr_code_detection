from fastapi import UploadFile, File
from PIL import Image
import io
from api.models import load_model, predict_with_model
from api.utils import decode_qr_code, convert_image_to_opencv
from api.schemas import QRPredictionResponse, BoundingBox

# Cargar el modelo
model = load_model()


async def predict_json(file: UploadFile = File(...)) -> QRPredictionResponse:
    """
    Recibe una imagen, realiza la detección de QR codes utilizando un modelo YOLOv8, y devuelve
    un JSON con las coordenadas de los bounding boxes y el contenido del QR code decodificado.

    Args:
        file (UploadFile): Imagen cargada por el usuario en formato JPEG, PNG, etc.

    Returns:
        QRPredictionResponse: Un objeto que contiene una lista de bounding boxes 
        (x_min, y_min, x_max, y_max) y el contenido del QR code decodificado dentro
        de cada bounding box.
    """
    # Leer la imagen recibida
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # Convertir la imagen a RGB si tiene un canal alfa (RGBA)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Convertir la imagen a formato OpenCV
    img = convert_image_to_opencv(image)

    # Realizar la predicción usando el modelo YOLOv8
    results = predict_with_model(model, img)

    # Extraer las coordenadas del bounding box y decodificar el QR
    predictions = []
    for result in results:
        for box in result.boxes:
            # Obtener las coordenadas de la caja [x_min, y_min, x_max, y_max]
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            
            # Decodificar el contenido del QR code en la región detectada
            qr_content = decode_qr_code(img, [x_min, y_min, x_max, y_max])
            
            # Crear el objeto BoundingBox usando el schema
            bbox = BoundingBox(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                qr_content=qr_content if qr_content else "No QR code content detected"
            )
            predictions.append(bbox)

    return QRPredictionResponse(predictions=predictions)

