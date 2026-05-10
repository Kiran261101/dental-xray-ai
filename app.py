import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.title("Dental X-ray Detection")

model = YOLO("runs/train/dental_xray_improved/weights/best.pt")

uploaded_file = st.file_uploader("Upload Dental X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width="stretch")
    
    file_bytes = uploaded_file.getvalue()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    results = model.predict(source=temp_path, save=False)

    result_image = results[0].plot()

    st.image(result_image, caption="Detection Result", width="stretch")

    os.remove(temp_path)

    print("test")