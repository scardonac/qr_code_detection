import streamlit as st
import requests
from PIL import Image
import io

# Definir la URL del backend (FastAPI)
API_URL_JSON = "http://127.0.0.1:8000/predict-json/"
API_URL_IMAGE = "http://127.0.0.1:8000/predict-image/"

# Configuración de la aplicación Streamlit
st.title("Detección de QR Code")
st.write("Sube una imagen para detectar códigos QR y extraer el enlace si se encuentra.")

# Subir la imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar la imagen original
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen original", use_column_width=True)

    # Convertir la imagen a bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format=image.format)
    image_bytes.seek(0)

    # Enviar la imagen para obtener el JSON con coordenadas y QR info
    if st.button("Obtener QR info"):
        files = {"file": (uploaded_file.name, image_bytes, "image/jpeg")}
        response = requests.post(API_URL_JSON, files=files)

        if response.status_code == 200:
            result = response.json()
            predictions = result["predictions"]
            if predictions:
                st.write("Códigos QR detectados:")
                for prediction in predictions:
                    st.write(f"QR Code Link: {prediction['qr_content']}")
            else:
                st.write("No se detectaron códigos QR.")
        else:
            st.write("Hubo un error en la predicción.")

    # Enviar la imagen para obtener la imagen con bounding boxes dibujados
    if st.button("Mostrar imagen con bounding boxes"):
        files = {"file": (uploaded_file.name, image_bytes, "image/jpeg")}
        response = requests.post(API_URL_IMAGE, files=files)

        if response.status_code == 200:
            img_data = response.content
            img_with_bboxes = Image.open(io.BytesIO(img_data))
            st.image(img_with_bboxes, caption="Imagen con Bounding Boxes", use_column_width=True)
        else:
            st.write("Hubo un error en la predicción.")
