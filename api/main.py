from fastapi import FastAPI, UploadFile, File
from api.predict import qr_detection
from api.schemas import QRPredictionResponse


app = FastAPI(
    title="QR Code Detection and Information Extraction API",
    description="""
    This API allows users to upload images and detect QR codes within them. 
    It returns the bounding boxes around the detected QR codes and extracts the content from the QR codes.

    ## Features:
    - Upload an image in various formats (JPG, PNG).
    - Detect QR codes and return bounding box coordinates.
    - Extract and return QR code content.
    - View bounding boxes drawn on the image for visual verification.
    """,
    version="1.0.0"
)


@app.post("/qr-detection/", response_model=QRPredictionResponse, tags=["QR Code Detection"])
async def qr_detection_endpoint(file: UploadFile = File(...)) -> QRPredictionResponse:
    return await qr_detection(file)

