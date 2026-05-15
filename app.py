import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
from collections import Counter

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Dental Xray AI Detection",
    layout="wide"
)

st.title("Dental Xray AI Detection Model")

# =========================
# LOAD MODELS
# =========================

@st.cache_resource
def load_models():
    strong_model = YOLO("models/strong_model.pt")
    weak_model = YOLO("models/weak_model.pt")
    return strong_model, weak_model

strong_model, weak_model = load_models()

# =========================
# FILE UPLOADER
# =========================

uploaded_file = st.file_uploader(
    "Upload Dental X-ray",
    type=["jpg", "jpeg", "png"]
)

# =========================
# CLASS COLORS
# =========================

color_map = {
    "CROWDING": (255, 0, 0),
    "CROWNS": (255, 255, 0),
    "ROTATED TOOTH": (0, 0, 255),
    "RCT TOOTH": (0, 255, 0),
    "IMPACTED TOOTH": (0, 255, 255),
    "SUPRA-ERUPTED TOOTH": (255, 0, 255),
    "BONE LOSS": (128, 255, 0),
    "ATTRITION": (255, 128, 0),
    "DENTAL CARIES": (255, 255, 255),
    "MISSING TOOTH": (255, 128, 128),
    "SPACING": (0, 128, 255),
    "PERIAPICAL INFECTION": (128, 0, 255),
    "ERUPTING TOOTH": (255, 0, 128),
    "ROOT STUMP": (0, 255, 255),
    "RESTORATIONS": (0, 255, 128),
}

# =========================
# PROCESS IMAGE
# =========================

if uploaded_file is not None:

    # READ IMAGE
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # SHOW RAW IMAGE
    st.subheader("Uploaded Raw Image")

    st.image(
        image_np,
        use_container_width=True
    )

    # COPY IMAGE
    final_img = image_np.copy()

    # =========================
    # STRONG MODEL PREDICTION
    # =========================

    strong_results = strong_model.predict(
        source=image_np,
        conf=0.10,
        imgsz=1280,
        verbose=False
    )[0]

    # =========================
    # WEAK MODEL PREDICTION
    # =========================

    weak_results = weak_model.predict(
        source=image_np,
        conf=0.10,
        imgsz=1280,
        verbose=False
    )[0]

    # =========================
    # SUMMARY COUNTER
    # =========================

    detected_classes = Counter()

    # =========================
    # COMBINE DETECTIONS
    # =========================

    all_boxes = []

    for box in strong_results.boxes:
        all_boxes.append(("strong", box))

    for box in weak_results.boxes:
        all_boxes.append(("weak", box))

    # =========================
    # DRAW DETECTIONS
    # =========================

    for model_type, box in all_boxes:

        # BOX COORDINATES
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # CONFIDENCE
        conf = float(box.conf[0])

        # CLASS ID
        cls = int(box.cls[0])

        # CLASS NAME
        if model_type == "strong":
            class_name = strong_model.names[cls].upper()
        else:
            class_name = weak_model.names[cls].upper()

        # CLASS COLOR
        color = color_map.get(class_name, (0, 255, 0))

        # =========================
        # CLEAN THIN BOX
        # =========================

        cv2.rectangle(
            final_img,
            (x1, y1),
            (x2, y2),
            color,
            1
        )

        # =========================
        # LABEL
        # =========================

        label = f"{class_name} {conf:.2f}"

        font_scale = 0.32
        thickness = 1

        # TEXT SIZE
        (text_width, text_height), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness
        )

        # =========================
        # LABEL BACKGROUND
        # =========================

        cv2.rectangle(
            final_img,
            (x1, y1 - 16),
            (x1 + text_width + 4, y1),
            color,
            -1
        )

        # =========================
        # LABEL TEXT
        # =========================

        cv2.putText(
            final_img,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )

        # =========================
        # SUMMARY COUNT
        # =========================

        detected_classes[class_name] += 1

    # =========================
    # SHOW FINAL OUTPUT
    # =========================

    st.subheader("Final Model Output")

    st.image(
        final_img,
        use_container_width=True
    )

    # =========================
    # DETECTION SUMMARY
    # =========================

    st.subheader("Detection Summary")

    if len(detected_classes) > 0:

        for cls_name, count in detected_classes.items():

            st.markdown(
                f"""
                <div style="
                    background-color:#1f4d2e;
                    padding:10px;
                    border-radius:10px;
                    margin-bottom:8px;
                    color:white;
                    font-weight:bold;
                    font-size:16px;
                ">
                    {cls_name} : {count}
                </div>
                """,
                unsafe_allow_html=True
            )

    else:
        st.warning("No detections found.")