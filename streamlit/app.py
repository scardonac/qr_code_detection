import streamlit as st
import requests
from PIL import Image, ImageDraw
import io


API_URL = "http://localhost:8000/predict-qr"

def draw_bounding_boxes(image: Image, predictions: list) -> Image:
    """
    Dibuja bounding boxes sobre una imagen basada en las predicciones proporcionadas.

    Args:
        image (PIL.Image): La imagen sobre la cual se dibujarán los bounding boxes.
        predictions (list): Una lista de predicciones, donde cada predicción es un 
        diccionario que contiene las coordenadas del bounding box.

    Returns:
        PIL.Image: La imagen con los bounding boxes dibujados en las áreas especificadas.
    """
    draw = ImageDraw.Draw(image)
    for prediction in predictions:
        bbox = prediction['x_min'], prediction['y_min'], prediction['x_max'], prediction['y_max']
        draw.rectangle(bbox, outline="red", width=3)
    return image


def main():
    """Main function of the app"""

    st.set_page_config(
        page_title="QR Detection", page_icon=":green_book:", layout="wide"
    )
    st.title("QR Code Detection")
    st.header(f"Upload an image to detect the QR code and get the information it contains 🤖")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Mostrar la imagen original
        image = Image.open(uploaded_file)
        st.image(image, caption='Original image', use_column_width=True)

        # Convertir la imagen a bytes para enviarla a la API
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes.seek(0)

        # Enviar la imagen a la API
        files = {"file": ("image.png", image_bytes, "image/png")}  # Asegurarse de enviar el archivo correctamente
        
        with st.spinner('Detecting QR codes...'):
            response = requests.post(API_URL, files=files)

        # Verificar si la respuesta fue exitosa
        if response.status_code == 200:
            # Procesar la respuesta JSON
            result = response.json()

            # Mostrar información de los QR codes
            st.subheader("QR Code Detection Results")
            predictions = result.get("predictions", [])

            if predictions:
                for idx, prediction in enumerate(predictions):
                    st.write(f"QR Code {idx + 1}:")
                    st.write(f"QR content: {prediction['qr_content']}")
            else:
                st.write("No QR codes were detected in the image.")

            # Agregar botón para mostrar la imagen con bounding boxes
            if st.button('Display image with Bounding Boxes'):
                # Dibujar los bounding boxes en la imagen original
                image_with_boxes = draw_bounding_boxes(image.copy(), predictions)
                # Mostrar la imagen con los bounding boxes
                st.image(image_with_boxes, caption="Image with Bounding Boxes", use_column_width=True)
        else:
            st.error(f"Error making prediction: {response.status_code}")


if __name__ == "__main__":
    main()