from fastapi import FastAPI, UploadFile, File
from api.predict import predict_json, predict_image
from api.schemas import QRPredictionResponse


app = FastAPI()

@app.post("/predict-json/", response_model=QRPredictionResponse)
async def predict_json_endpoint(file: UploadFile = File(...)) -> QRPredictionResponse:
    return await predict_json(file)

@app.post("/predict-image/")
async def predict_image_endpoint(file: UploadFile = File(...)):
    return await predict_image(file)
