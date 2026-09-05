"""Create a deterministic low-visibility test protocol from held-out Aquarium images."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets" / "aquarium-fish-yolo"
TARGET = ROOT / "datasets" / "aquarium-fish-robust-yolo"
TARGET_CLAHE = ROOT / "datasets" / "aquarium-fish-robust-clahe-yolo"


def degrade(image: np.ndarray) -> np.ndarray:
    """Simulate underwater attenuation, haze and slight motion blur deterministically."""
    image = cv2.GaussianBlur(image, (3, 3), 0).astype(np.float32) / 255.0
    b, g, r = cv2.split(image)
    attenuated = cv2.merge((np.clip(b * 0.88 + 0.10, 0, 1), np.clip(g * 0.75 + 0.08, 0, 1), np.clip(r * 0.52 + 0.04, 0, 1)))
    haze = np.full_like(attenuated, (0.56, 0.64, 0.69))
    degraded = attenuated * 0.72 + haze * 0.28
    return np.clip(degraded * 255.0, 0, 255).astype(np.uint8)


def clahe(image: np.ndarray) -> np.ndarray:
    """Apply the same CLAHE parameters used for the training variant."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def main() -> None:
    """Write test-only degraded and degraded-plus-CLAHE protocols with unchanged labels."""
    source_images = SOURCE / "images" / "test"
    source_labels = SOURCE / "labels" / "test"
    for root in (TARGET, TARGET_CLAHE):
        (root / "images" / "test").mkdir(parents=True, exist_ok=True)
        (root / "labels" / "test").mkdir(parents=True, exist_ok=True)

    count = 0
    for source in source_images.iterdir():
        if not source.is_file():
            continue
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unable to read image: {source}")
        degraded = degrade(image)
        outputs = ((TARGET, degraded), (TARGET_CLAHE, clahe(degraded)))
        for root, output in outputs:
            target = root / "images" / "test" / source.name
            if not cv2.imwrite(str(target), output, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"Unable to write image: {target}")
            shutil.copy2(source_labels / f"{source.stem}.txt", root / "labels" / "test" / f"{source.stem}.txt")
        count += 1
    print(f"created {count} degraded test images")


if __name__ == "__main__":
    main()
