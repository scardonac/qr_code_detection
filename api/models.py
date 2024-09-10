from ultralytics import YOLO
from api.config import MODEL_PATH
import numpy as np


def load_model() -> YOLO:
    """
    Carga un modelo YOLO preentrenado desde la ruta especificada en MODEL_PATH.

    Returns:
        YOLO: El modelo YOLO cargado listo para realizar predicciones.
    """
    model = YOLO(MODEL_PATH)
    return model


def predict_with_model(model: YOLO, 
                       image: np.ndarray) -> YOLO:
    """
    Realiza una predicción sobre una imagen utilizando un modelo YOLO cargado.

    Args:
        model (YOLO): El modelo YOLO previamente cargado.
        image (np.ndarray): La imagen en formato NumPy sobre la que se realizará la predicción.

    Returns:
        YOLO: El objeto de resultados que contiene las predicciones de la imagen, 
        incluyendo las cajas delimitadoras y las clases.
    """
    results = model.predict(source=image, save=False) 
    return results
