import streamlit as st
import requests
from PIL import Image, ImageDraw
import io


API_URL = "http://localhost:8000/qr-detection"


def draw_bounding_boxes(image: Image, predictions: list) -> Image:
    """
    Draws bounding boxes on an image based on the given predictions.

    Args:
        image (PIL.Image): The image on which the bounding boxes will be drawn.
        predictions (list): A list of predictions, where each prediction is a
        dictionary containing the coordinates of the bounding box.

    Returns:
        PIL.Image: The image with the bounding boxes drawn in the specified areas.
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
        # Show original image
        image = Image.open(uploaded_file)
        st.image(image, caption='Original image', use_column_width=True)

        # Convert the image to bytes to send to the API
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes.seek(0)

        files = {"file": ("image.png", image_bytes, "image/png")}
        
        with st.spinner('Detecting QR codes...'):
            response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            result = response.json()
            st.subheader("QR Code Detection Results")
            predictions = result.get("predictions", [])

            if predictions:
                for idx, prediction in enumerate(predictions):
                    st.write(f"QR Code {idx + 1}:")
                    st.write(f"QR content: {prediction['qr_content']}")
            else:
                st.write("No QR codes were detected in the image.")

            if st.button('Display image with Bounding Boxes'):
                image_with_boxes = draw_bounding_boxes(image.copy(), predictions)
                st.image(image_with_boxes, caption="Image with Bounding Boxes", use_column_width=True)
        else:
            st.error(f"Error making prediction: {response.status_code}")


if __name__ == "__main__":
    main()