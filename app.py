import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
from collections import Counter

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Dental Xray AI Detection",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("Dental Xray AI Detection Model")

# ---------------- CLASS COLORS ----------------
CLASS_COLORS = {
    "CROWDING": (255, 120, 1200),
    "CROWNS": (120, 255, 255),
    "ROTATED": (220, 120, 255),
    "RCT TOOTH": (120, 255, 180),
    "IMPACTED TOOTH": (255, 220, 120),
    "SUPRA ERUPTED TOOTH": (255, 170, 120),

    "BONE LOSS": (120, 180, 255),
    "ATTRITION": (180, 120, 255),
    "DENTAL CARIES": (245, 245, 245),
    "MISSING TOOTH": (120, 150, 255),
    "SPACING": (200, 255, 120),
    "PERIAPICAL INFECTION": (255, 120, 180),
    "ERUPTING TOOTH": (180, 150, 255),
    "ROOT STUMP": (220, 220, 120),
    "RESTORATIONS": (120, 255, 120),
}

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    strong_model = YOLO("models/strong_model.pt")
    weak_model = YOLO("models/weak_model.pt")
    return strong_model, weak_model

strong_model, weak_model = load_models()

# ---------------- FILE UPLOADER ----------------
uploaded_file = st.file_uploader(
    "Upload Dental Xray",
    type=["png", "jpg", "jpeg"]
)

# ---------------- PROCESS ----------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.subheader("Uploaded Raw Image")
    st.image(
        image_np,
        use_container_width=True
    )

    # Save temp image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_path = temp_file.name
        image.save(temp_path)

    # ---------------- PREDICTIONS ----------------
    strong_results = strong_model.predict(
        source=temp_path,
        conf=0.20,
        verbose=False
    )

    weak_results = weak_model.predict(
        source=temp_path,
        conf=0.20,
        verbose=False
    )

    # ---------------- FINAL IMAGE ----------------
    final_img = image_np.copy()

    detected_classes = []

    def draw_predictions(results, model):

        global final_img

        for box in results[0].boxes:

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            class_name = model.names[cls_id].upper()

            detected_classes.append(class_name)

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = CLASS_COLORS.get(class_name, (255, 255, 255))

           # DRAW THINNER BOX
            cv2.rectangle(
                final_img,
                (x1, y1),
                (x2, y2),
                color,
                1
            )

            # SIMPLE YOLO STYLE LABEL
            label = f"{class_name} {conf:.2f}"

            cv2.putText(
                final_img,
                label,
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA
            )

    # DRAW BOTH MODELS
    draw_predictions(strong_results, strong_model)
    draw_predictions(weak_results, weak_model)

    # ---------------- SHOW FINAL IMAGE ----------------
    st.subheader("Final Model")

    st.image(
        final_img,
        use_container_width=True
    )

    # ---------------- DETECTION SUMMARY ----------------
    st.markdown("---")
    st.subheader("Detection Summary")

    class_counter = Counter(detected_classes)

    if class_counter:

        cols = st.columns(3)

        for idx, (cls, count) in enumerate(sorted(class_counter.items())):

            with cols[idx % 3]:
                st.success(f"{cls} : {count}")

    else:
        st.warning("No detections found.")