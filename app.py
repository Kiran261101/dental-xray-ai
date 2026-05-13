import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
import numpy as np

st.set_page_config(layout="wide")

st.title("Dental X-ray Model Comparison")

# LOAD MODELS
old_model = YOLO("models/old_model.pt")
strong_model = YOLO("models/strong_model.pt")
weak_model = YOLO("models/weak_model.pt")

# UPLOAD IMAGE
uploaded_file = st.file_uploader(
    "Upload Dental X-ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # SHOW RAW IMAGE
    image = Image.open(uploaded_file)

    st.subheader("Uploaded Raw Image")
    st.image(image, use_container_width=True)

    # SAVE TEMP FILE
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    # OLD MODEL PREDICTION
    old_results = old_model.predict(
        source=temp_path,
        conf=0.4
    )

    old_image = old_results[0].plot(
        line_width=5,
        font_size=20
    )

    # STRONG MODEL PREDICTION
    strong_results = strong_model.predict(
        source=temp_path,
        conf=0.4
    )

    strong_image = strong_results[0].plot(
        line_width=5,
        font_size=20
    )

    # WEAK MODEL PREDICTION
    weak_results = weak_model.predict(
        source=temp_path,
        conf=0.4
    )

    weak_image = weak_results[0].plot(
        line_width=5,
        font_size=20
    )

    # CONVERT BGR TO RGB
    old_image = cv2.cvtColor(old_image, cv2.COLOR_BGR2RGB)
    strong_image = cv2.cvtColor(strong_image, cv2.COLOR_BGR2RGB)
    weak_image = cv2.cvtColor(weak_image, cv2.COLOR_BGR2RGB)

    # DISPLAY SIDE BY SIDE
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Old Model")
        st.image(old_image, use_container_width=True)

    with col2:
        st.subheader("New Combined Model")

        # SHOW STRONG RESULT
        st.image(strong_image, caption="Strong Model", use_container_width=True)

        # SHOW WEAK RESULT
        st.image(weak_image, caption="Weak Model", use_container_width=True)