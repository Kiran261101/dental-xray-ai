import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
import numpy as np

st.title("Dental X-ray Model Comparison")

# Load both models
old_model = YOLO("runs/detect/train7/weights/best.pt")
new_model = YOLO("runs/detect/train8/weights/best.pt")

uploaded_file = st.file_uploader(
    "Upload Dental X-ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Original Image", width=500)

    # Convert uploaded image into temp file
    file_bytes = uploaded_file.getvalue()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    # OLD MODEL PREDICTION
    old_results = old_model.predict(source=temp_path, save=False)
    old_image = old_results[0].plot()

    # NEW MODEL PREDICTION
    new_results = new_model.predict(source=temp_path, save=False)
    new_image = new_results[0].plot()

    # Convert BGR to RGB
    old_image = cv2.cvtColor(old_image, cv2.COLOR_BGR2RGB)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)

    # Side-by-side display
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Old Model ")
        st.image(old_image)

    with col2:
        st.subheader("New Model ")
        st.image(new_image)