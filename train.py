import os
import logging
import argparse
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_files(data_path, model_path):
    """Validate that required files exist."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    logger.info("All required files validated successfully.")

def train_model(data_path, model_path, epochs, imgsz, batch, device):
    """Train the YOLO model with given parameters."""
    try:
        logger.info("Loading YOLO model...")
        model = YOLO(model_path)

        logger.info("Starting training...")
        results = model.train(
            data=data_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device
        )
        logger.info("Training completed successfully.")
        return results
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

def main():
    model = YOLO("yolo11n.pt")

    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0
    )

if __name__ == "__main__":
    main()