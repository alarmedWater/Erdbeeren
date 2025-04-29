#!/usr/bin/env python3
"""
Script to generate YOLO labels, split dataset, and train a YOLOv8 model.
"""
import argparse
import shutil
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

def generate_labels(image_dir: Path, label_dir: Path, class_id: int = 0) -> int:
    """
    Generate YOLO-format labels (full-image boxes) for all images in `image_dir`.
    """
    image_dir = image_dir.resolve()
    label_dir = label_dir.resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory {image_dir} does not exist")
    label_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for img in image_dir.iterdir():
        if img.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            label_file = label_dir / f"{img.stem}.txt"
            with label_file.open('w') as f:
                # Full-image bounding box: x_center=0.5, y_center=0.5, width=1.0, height=1.0
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
            count += 1

    logging.info(f"✅ {count} labels created in {label_dir}")
    return count


def split_dataset(
    img_src: Path,
    lbl_src: Path,
    img_train: Path,
    img_val: Path,
    lbl_train: Path,
    lbl_val: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    extensions: set = None
) -> tuple[list[str], list[str]]:
    """
    Split images and labels into training and validation sets.
    """
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png'}

    # Create directories
    for d in (img_train, img_val, lbl_train, lbl_val):
        d.mkdir(parents=True, exist_ok=True)

    # Collect stems
    stems = [f.stem for f in img_src.iterdir() if f.suffix.lower() in extensions]
    train_stems, val_stems = train_test_split(stems, test_size=test_size, random_state=random_state)

    # Copy files
    def copy_files(stems: list[str], dst_img: Path, dst_lbl: Path):
        for stem in stems:
            for ext in extensions:
                src_img = img_src / f"{stem}{ext}"
                if src_img.exists():
                    shutil.copy(src_img, dst_img / src_img.name)
                    break
            shutil.copy(lbl_src / f"{stem}.txt", dst_lbl / f"{stem}.txt")

    copy_files(train_stems, img_train, lbl_train)
    copy_files(val_stems, img_val, lbl_val)

    logging.info(f"Train: {len(train_stems)} | Val: {len(val_stems)}")
    return train_stems, val_stems


def train_yolo(
    data_config: str,
    weights: str = 'yolov8n.pt',
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    augment: bool = True,
    project: str = 'runs/train',
    name: str = 'exp'
) -> None:
    """
    Train a YOLOv8 model using the Ultralytics API.
    """
    model = YOLO(weights)
    model.train(
        data=data_config,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        augment=augment,
        project=project,
        name=name
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO Data Prep & Training Pipeline"
    )
    parser.add_argument(
        '--base-dir', type=Path, default=Path().resolve(),
        help='Base directory for data and outputs'
    )
    parser.add_argument('--class-id', type=int, default=0, help='Class ID for all labels')
    parser.add_argument('--test-size', type=float, default=0.2, help='Fraction for validation set')
    parser.add_argument('--random-state', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='Pretrained weights file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Training image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument('--project', type=str, default='runs/train', help='Training project directory')
    parser.add_argument('--name', type=str, default='ripe_only', help='Training run name')
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    base = args.base_dir
    img_dir = base / 'Data' / 'Normal' / 'Ripe'
    lbl_dir = base / 'labels' / 'Ripe'

    # Step 1: Generate labels
    generate_labels(img_dir, lbl_dir, class_id=args.class_id)

    # Step 2: Split dataset
    img_train = base / 'images' / 'train'
    img_val   = base / 'images' / 'val'
    lbl_train = base / 'labels' / 'train'
    lbl_val   = base / 'labels' / 'val'
    split_dataset(
        img_dir, lbl_dir,
        img_train, img_val,
        lbl_train, lbl_val,
        test_size=args.test_size,
        random_state=args.random_state
    )

    # Step 3: Train YOLO
    train_yolo(
        data_config='data.yaml',
        weights=args.weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        augment=args.augment,
        project=args.project,
        name=args.name
    )

if __name__ == '__main__':
    main()
