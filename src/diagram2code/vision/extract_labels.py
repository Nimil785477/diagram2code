from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import os
import shutil

import cv2

from diagram2code.schema import Node


def _configure_tesseract_cmd(pytesseract) -> None:
    """
    Make pytesseract find the tesseract binary in a cross-platform way.
    - If TESSERACT_CMD env var is set, use it.
    - Else if `tesseract` is on PATH, use it.
    - Else on Windows, try common install locations.
    - Else leave as-is (pytesseract will raise TesseractNotFoundError).
    """
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd:
        pytesseract.pytesseract.tesseract_cmd = env_cmd
        return

    which = shutil.which("tesseract")
    if which:
        pytesseract.pytesseract.tesseract_cmd = which
        return

    # Common Windows locations (only if needed)
    if os.name == "nt":
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files\Tesseract-OCR\tesseract",
        ]
        for c in candidates:
            if Path(c).exists():
                pytesseract.pytesseract.tesseract_cmd = c
                return


def extract_node_labels(bgr_img, nodes: List[Node]) -> Dict[int, str]:
    """
    OCR each node rectangle region and return {node_id: text}.
    If pytesseract/tesseract is unavailable, returns {} gracefully.
    """
    try:
        import pytesseract
        from pytesseract import TesseractNotFoundError
    except ImportError:
        return {}

    _configure_tesseract_cmd(pytesseract)

    labels: Dict[int, str] = {}

    for n in nodes:
        x, y, w, h = n.bbox
        roi = bgr_img[y : y + h, x : x + w]
        if roi.size == 0:
            continue

        # Light preprocessing for OCR
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        try:
            text = pytesseract.image_to_string(gray, config="--psm 6").strip()
        except TesseractNotFoundError:
            return {}
        except Exception:
            continue

        if text:
            labels[n.id] = text

    return labels
