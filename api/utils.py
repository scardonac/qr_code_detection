import cv2
import numpy as np
from pyzbar.pyzbar import decode
from PIL import Image


def decode_qr_code(image: np.ndarray, bbox: list) -> str:
    """
    Extracts the region of the image delimited by the bounding box and decodes the QR code.

    Args:
        image (np.ndarray): The full image in OpenCV format (NumPy array).
        bbox (list): List containing the coordinates [x_min, y_min, x_max, y_max] of the
        bounding box that delimits the QR code.

    Returns:
        str: The content of the QR code if it is decoded correctly, or None if no
        QR code is detected.
    """
    x_min, y_min, x_max, y_max = bbox
    qr_region = image[y_min:y_max, x_min:x_max]
    
    decoded_objects = decode(qr_region)
    
    if decoded_objects:
        qr_data = decoded_objects[0].data.decode('utf-8')
        return qr_data
    else:
        return None


def convert_image_to_opencv(image: Image) -> np.ndarray:
    """
    Converts a PIL image to OpenCV format (NumPy array). If the image is in RGB mode,
    converts the color channels from RGB to BGR for compatibility with OpenCV.

    Args:
        image (Image): The image in PIL format.

    Returns:
        np.ndarray: The image converted to OpenCV format (NumPy array) with color channels
        BGR if RGB.
    """
    img = np.array(image)
    if image.mode == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img