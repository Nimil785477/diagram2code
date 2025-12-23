# src/diagram2code/vision/extract_labels.py

from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import os
import shutil

import cv2
import pytesseract

from diagram2code.schema import Node


def _configure_tesseract() -> None:
    """
    Make pytesseract find the tesseract binary across OSes.
    Priority:
      1) env var TESSERACT_CMD
      2) tesseract on PATH
      3) common Windows install locations
    """
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd and Path(env_cmd).exists():
        pytesseract.pytesseract.tesseract_cmd = env_cmd
        return

    which = shutil.which("tesseract")
    if which:
        pytesseract.pytesseract.tesseract_cmd = which
        return

    # Common Windows locations
    candidates = [
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ]
    for c in candidates:
        if c.exists():
            pytesseract.pytesseract.tesseract_cmd = str(c)
            return


_configure_tesseract()


def extract_node_labels(image_bgr, nodes: List[Node]) -> Dict[int, str]:
    labels: Dict[int, str] = {}

    for n in nodes:
        x, y, w, h = n.bbox
        roi = image_bgr[y : y + h, x : x + w]
        if roi.size == 0:
            continue

        text = pytesseract.image_to_string(roi, config="--psm 6").strip()
        if text:
            labels[n.id] = text

    return labels
