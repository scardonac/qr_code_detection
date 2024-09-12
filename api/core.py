from ultralytics import YOLO
import numpy as np


class QRCodeDetector:
    """
    Class that encapsulates the loading logic of a YOLO model and QR code predictions on images.
    """

    def __init__(self, model_path: str):
        """
        Initializes the QR code detector with the path of the pre-trained YOLO model.

        Args:
            model_path (str): The path to the pretrained YOLO model file.
        """
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self) -> YOLO:
        """
        Loads the pre-trained YOLO model from the path specified when initializing the class.

        Returns:
            YOLO: The YOLO model loaded ready to make predictions.
        """
        model = YOLO(self.model_path)
        return model

    def predict(self, image: np.ndarray) -> YOLO:
        """
        Make a prediction on an image using the loaded YOLO model.

        Args:
            image (np.ndarray): The image in NumPy format on which the prediction will be made.

        Returns:
            YOLO: The result object containing the predictions for the image,
            including bounding boxes and classes.
        """
        results = self.model.predict(source=image, save=False)
        return results
