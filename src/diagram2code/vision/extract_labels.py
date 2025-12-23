# src/diagram2code/vision/extract_labels.py

from __future__ import annotations
from typing import Dict, List
from pathlib import Path

import cv2
import pytesseract

from diagram2code.schema import Node


# ✅ HARD-CODE TESSERACT PATH (Windows-safe)
if pytesseract.pytesseract.tesseract_cmd == "tesseract":
    candidate = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if candidate.exists():
        pytesseract.pytesseract.tesseract_cmd = str(candidate)


def extract_node_labels(image_bgr, nodes: List[Node]) -> Dict[int, str]:
    labels: Dict[int, str] = {}

    for n in nodes:
        x, y, w, h = n.bbox
        roi = image_bgr[y:y+h, x:x+w]

        if roi.size == 0:
            continue

        text = pytesseract.image_to_string(
            roi,
            config="--psm 6",
        ).strip()

        if text:
            labels[n.id] = text

    return labels
