from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessResult:
    original_path: Path
    output_path: Path
    image_gray: np.ndarray
    image_bin: np.ndarray


def preprocess_bgr_to_bin(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Pure preprocessing: takes a BGR image array and returns (gray, bin_img).
    Does NOT write anything to disk. Safe to use in --dry-run.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Robust to uneven lighting (phone photos)
    bin_img = cv2.adaptiveThreshold(
        gray_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )

    # Close tiny gaps in strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)

    return gray, bin_img


def preprocess_image(input_path: str | Path, out_dir: str | Path) -> PreprocessResult:
    """
    File-based preprocessing: reads the image, computes binarization,
    and writes preprocessed.png to out_dir.
    """
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(input_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    gray, bin_img = preprocess_bgr_to_bin(img)

    out_path = out_dir / "preprocessed.png"
    cv2.imwrite(str(out_path), bin_img)

    return PreprocessResult(
        original_path=input_path,
        output_path=out_path,
        image_gray=gray,
        image_bin=bin_img,
    )
