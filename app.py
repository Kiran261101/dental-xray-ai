import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
import numpy as np

# PAGE SETTINGS
st.set_page_config(layout="wide")

# TITLE
st.title("Dental X-ray Model Comparison")

# LOAD MODELS
old_model = YOLO("models/old_model.pt")
strong_model = YOLO("models/strong_model.pt")
weak_model = YOLO("models/weak_model.pt")

# FILE UPLOADER
uploaded_file = st.file_uploader(
    "Upload Dental X-ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # OPEN IMAGE
    image = Image.open(uploaded_file).convert("RGB")

    # SHOW RAW IMAGE
    st.subheader("Uploaded Raw Image")
    st.image(image, use_container_width=True)

    # SAVE TEMP IMAGE
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    # ---------------- OLD MODEL ----------------
    old_results = old_model.predict(
        source=temp_path,
        conf=0.2
    )

    old_image = old_results[0].plot(
        line_width=2,
        font_size=8
    )

    # ---------------- STRONG MODEL ----------------
    strong_results = strong_model.predict(
        source=temp_path,
        conf=0.2
    )

    # ---------------- WEAK MODEL ----------------
    weak_results = weak_model.predict(
        source=temp_path,
        conf=0.2
    )

    # CREATE COMBINED IMAGE
    combined_image = np.array(image).copy()

    # DRAW STRONG MODEL DETECTIONS
    strong_drawn = strong_results[0].plot(
        img=combined_image,
        line_width=2,
        font_size=8
    )

    # DRAW WEAK MODEL DETECTIONS ON SAME IMAGE
    combined_final = weak_results[0].plot(
        img=strong_drawn,
        line_width=2,
        font_size=8
    )

    # CONVERT COLORS
    old_image = cv2.cvtColor(old_image, cv2.COLOR_BGR2RGB)
    combined_final = cv2.cvtColor(combined_final, cv2.COLOR_BGR2RGB)

    # DISPLAY SIDE BY SIDE
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Old Model")
        st.image(old_image, use_container_width=True)

    with col2:
        st.subheader("New Combined Model")
        st.image(combined_final, use_container_width=True)