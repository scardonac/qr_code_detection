from fastapi import UploadFile, File
from PIL import Image
import io
from api.core import QRCodeDetector
from api.utils import decode_qr_code, convert_image_to_opencv
from api.config import MODEL_PATH


async def qr_detection(file: UploadFile = File(...)) -> dict:
    """
    Receives an image, performs QR code detection using a YOLOv8 model, and returns
    a JSON with the bounding box coordinates and decoded QR code content.

    Args:
        file (UploadFile): Image uploaded by the user in JPEG, PNG, etc.

    Returns:
        QRPredictionResponse: An object containing a list of bounding boxes 
        (x_min, y_min, x_max, y_max) and the decoded QR code content within 
        each bounding box.
    """

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # Convert image to RGB if it has an alpha channel (RGBA)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Convert the image to OpenCV format
    img = convert_image_to_opencv(image)

    # Load the model
    detector = QRCodeDetector(model_path=MODEL_PATH)

    # Perform prediction using the YOLOv8 model
    results = detector.predict(image)

    predictions = []

    for result in results:
        for box in result.boxes:
            # Get the bounding box coordinates [x_min, y_min, x_max, y_max]
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])

            # Decode the QR code content within the detected bounding box region
            qr_content = decode_qr_code(img, [x_min, y_min, x_max, y_max])

            predictions.append({
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
                'qr_content': qr_content if qr_content else "No QR code content detected"
            })

    return {"predictions": predictions}

