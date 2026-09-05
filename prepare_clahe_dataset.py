"""Create an image-level CLAHE variant of a YOLO detection dataset."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets" / "aquarium-fish-yolo"
TARGET = ROOT / "datasets" / "aquarium-fish-clahe-yolo"
SPLITS = ("train", "val", "test")


def enhance_image(source: Path, target: Path) -> None:
    """Apply CLAHE on the LAB luminance channel and write a JPEG with the original filename."""
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to read image: {source}")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    if not cv2.imwrite(str(target), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Unable to write image: {target}")


def main() -> None:
    """Generate enhanced images and copy the matching YOLO labels."""
    for split in SPLITS:
        src_images = SOURCE / "images" / split
        src_labels = SOURCE / "labels" / split
        dst_images = TARGET / "images" / split
        dst_labels = TARGET / "labels" / split
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)
        count = 0
        for image_path in src_images.iterdir():
            if not image_path.is_file():
                continue
            enhance_image(image_path, dst_images / image_path.name)
            label_path = src_labels / f"{image_path.stem}.txt"
            shutil.copy2(label_path, dst_labels / label_path.name)
            count += 1
        print(f"{split}: {count} images")


if __name__ == "__main__":
    main()
