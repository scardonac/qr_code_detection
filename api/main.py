from fastapi import FastAPI, UploadFile, File
from api.predict import predict_json
from api.schemas import QRPredictionResponse


app = FastAPI()

@app.post("/predict-qr/", response_model=QRPredictionResponse)
async def predict_json_endpoint(file: UploadFile = File(...)) -> QRPredictionResponse:
    return await predict_json(file)

