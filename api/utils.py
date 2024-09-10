import cv2
import numpy as np
from pyzbar.pyzbar import decode
from PIL import Image


def decode_qr_code(image: np.ndarray, bbox: list) -> str:
    """
    Extrae la región de la imagen delimitada por el bounding box y decodifica el código QR.

    Args:
        image (np.ndarray): La imagen completa en formato OpenCV (NumPy array).
        bbox (list): Lista que contiene las coordenadas [x_min, y_min, x_max, y_max] del
        bounding box que delimita el QR code.

    Returns:
        str: El contenido del código QR si es decodificado correctamente, o None si no se
        detecta un QR code.
    """
    x_min, y_min, x_max, y_max = bbox
    qr_region = image[y_min:y_max, x_min:x_max]
    
    # Usar pyzbar para decodificar la región QR
    decoded_objects = decode(qr_region)
    
    # Extraer contenido si es decodificado
    if decoded_objects:
        qr_data = decoded_objects[0].data.decode('utf-8')
        return qr_data
    else:
        return None


def convert_image_to_opencv(image: Image) -> np.ndarray:
    """
    Convierte una imagen PIL a formato OpenCV (NumPy array). Si la imagen está en modo RGB,
    convierte los canales de color de RGB a BGR para compatibilidad con OpenCV.

    Args:
        image (Image): La imagen en formato PIL.

    Returns:
        np.ndarray: La imagen convertida a formato OpenCV (NumPy array) con canales de color
        BGR si es RGB.
    """
    img = np.array(image)
    if image.mode == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img