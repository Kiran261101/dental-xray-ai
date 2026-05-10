import os
import logging
import argparse
from collections import Counter
from ultralytics import YOLO

try:
    import torch
except ImportError:
    torch = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_default_device() -> str:
    if torch is not None and torch.cuda.is_available():
        logger.info('CUDA-enabled GPU detected, using device 0')
        return '0'
    logger.warning('GPU not available, falling back to CPU')
    return 'cpu'



def validate_files(data_path: str, model_path: str) -> None:
    """Validate that required files exist."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    logger.info("All required files validated successfully.")


def summarize_dataset(data_path: str) -> None:
    """Summarize label file counts and class frequencies."""
    logger.info("Summarizing dataset using %s", data_path)
    base_dir = os.path.dirname(data_path)
    labels_dir = os.path.join(base_dir, 'labels')

    for split in ['train', 'val']:
        split_path = os.path.join(labels_dir, split)
        if not os.path.isdir(split_path):
            logger.warning('Missing label directory for split: %s', split_path)
            continue

        file_count = len([name for name in os.listdir(split_path) if name.endswith('.txt')])
        counts = Counter()
        total_boxes = 0

        for label_file in os.listdir(split_path):
            if not label_file.endswith('.txt'):
                continue
            with open(os.path.join(split_path, label_file), 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    counts[parts[0]] += 1
                    total_boxes += 1

        average = total_boxes / file_count if file_count else 0
        logger.info('%s split: %d label files, %d boxes total, %.1f boxes/image', split, file_count, total_boxes, average)
        for cls, count in counts.most_common(8):
            logger.info('  class %s: %d boxes', cls, count)


def train_model(
    data_path: str,
    model_path: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    lr0: float,
    lrf: float,
    optimizer: str,
    patience: int,
    cache: str,
    augment: bool,
    workers: int,
    project: str,
    name: str,
) -> None:
    """Train the YOLO model with better default settings."""
    validate_files(data_path, model_path)
    summarize_dataset(data_path)

    logger.info('Loading YOLO weights from %s', model_path)
    model = YOLO(model_path)

    logger.info('Training with epochs=%d, imgsz=%d, batch=%d, optimizer=%s, workers=%d', epochs, imgsz, batch, optimizer, workers)
    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        lr0=lr0,
        lrf=lrf,
        optimizer=optimizer,
        patience=patience,
        cache=cache,
        augment=augment,
        workers=workers,
        project=project,
        name=name,
        exist_ok=True,
    )

    logger.info('Training completed successfully. Results saved to %s/%s', project, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a YOLO model for dental x-ray detection.')
    parser.add_argument('--data', default='data.yaml', help='Path to dataset YAML file.')
    parser.add_argument('--model', default='yolo26n.pt', help='Pretrained model weights path.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--imgsz', type=int, default=640, help='Training image size.')
    parser.add_argument('--batch', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--device', default=get_default_device(), help='Device to use for training.')
    parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--lrf', type=float, default=0.01, help='Final learning rate factor (lrf).')
    parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam', 'AdamW'], help='Optimizer type.')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience.')
    parser.add_argument('--cache', default='disk', choices=['ram', 'disk', 'none'], help='Dataset caching mode.')
    parser.add_argument('--augment', action='store_true', default=True, help='Enable built-in augmentation.')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers for data loading.')
    parser.add_argument('--project', default='runs/train', help='Project folder for saving results.')
    parser.add_argument('--name', default='dental_xray_improved', help='Name for the training run.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(
        data_path=args.data,
        model_path=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        lr0=args.lr0,
        lrf=args.lrf,
        optimizer=args.optimizer,
        patience=args.patience,
        cache=args.cache,
        augment=args.augment,
        workers=args.workers,
        project=args.project,
        name=args.name,
    )


if __name__ == '__main__':
    main()
