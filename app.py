import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

st.title("Dental X-ray Model Comparison")

# =========================
# LOAD MODELS
# =========================

old_model = YOLO("models/old_model.pt")

strong_model = YOLO("models/strong_model.pt")

weak_model = YOLO("models/weak_model.pt")

# =========================
# FILE UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload Dental X-ray",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PREDICTION
# =========================

if uploaded_file is not None:
    
    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    image_np = np.array(image)

    # =========================
    # OLD MODEL
    # =========================

    old_results = old_model.predict(image_np)

    old_image = old_results[0].plot()

    old_image = cv2.cvtColor(old_image, cv2.COLOR_BGR2RGB)

    # =========================
    # STRONG + WEAK MODEL
    # =========================

    strong_results = strong_model.predict(image_np)

    weak_results = weak_model.predict(image_np)

    # Copy original image
    combined_image = image_np.copy()

    # Draw STRONG boxes
    for box in strong_results[0].boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        conf = float(box.conf[0])

        cls = int(box.cls[0])

        label = strong_model.names[cls]

        cv2.rectangle(combined_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            combined_image,
            f"{label} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # Draw WEAK boxes
    for box in weak_results[0].boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        conf = float(box.conf[0])

        cls = int(box.cls[0])

        label = weak_model.names[cls]

        cv2.rectangle(combined_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.putText(
            combined_image,
            f"{label} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )

    # Convert BGR -> RGB
    combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)

    # =========================
    # DISPLAY
    # =========================

    # =========================
    # RAW IMAGE
    # =========================

    raw_image = image_np.copy()

    # =========================
    # COLOR MAP
    # =========================

    class_colors = {
        "CROWDING": (255, 0, 0),
        "CROWNS": (0, 255, 0),
        "ROTATED": (0, 0, 255),
        "RCT TOOTH": (255, 255, 0),
        "IMPACTED TOOTH": (255, 0, 255),
        "SUPRAERUPTED TOOTH": (0, 255, 255),

        "BONE LOSS": (128, 0, 255),
        "ATTRITION": (255, 128, 0),
        "DENTAL CARIES": (0, 128, 255),
        "MISSING TOOTH": (128, 255, 0),
        "SPACING": (255, 0, 128),
        "PERIAPICAL INFECTION": (0, 255, 128),
        "ERUPTING TOOTH": (128, 128, 255),
        "ROOT STUMP": (255, 128, 128),
        "RESTORATIONS": (128, 255, 255),
    }

    # =========================
    # DRAW STRONG BOXES
    # =========================

    for box in strong_results[0].boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        conf = float(box.conf[0])

        cls = int(box.cls[0])

        label = strong_model.names[cls]

        color = class_colors.get(label, (0, 255, 0))

        cv2.rectangle(combined_image, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            combined_image,
            f"{label} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    # =========================
    # DRAW WEAK BOXES
    # =========================

    for box in weak_results[0].boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        conf = float(box.conf[0])

        cls = int(box.cls[0])

        label = weak_model.names[cls]

        color = class_colors.get(label, (255, 255, 255))

        cv2.rectangle(combined_image, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            combined_image,
            f"{label} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    # =========================
    # CONVERT COLORS
    # =========================

    combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)

    old_image = cv2.cvtColor(old_image, cv2.COLOR_BGR2RGB)

    # =========================
    # DISPLAY
    # =========================

    st.subheader("Uploaded Raw Image")
    st.image(raw_image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Old Model")
        st.image(old_image)

    with col2:
        st.subheader("New Combined Model")
        st.image(combined_image)
