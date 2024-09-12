from pydantic import BaseModel
from typing import List, Optional


class QRBoundingBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    qr_content: Optional[str] = "No QR code content detected"


class QRPredictionResponse(BaseModel):
    predictions: List[QRBoundingBox]
