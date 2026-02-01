from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2

from diagram2code.schema import Node


def _configure_tesseract_cmd(pytesseract) -> None:
    """
    Configure pytesseract to find the tesseract binary cross-platform.

    Priority:
    1) TESSERACT_CMD env var (explicit)
    2) tesseract on PATH
    3) common Windows install locations
    Otherwise: leave default and pytesseract will raise TesseractNotFoundError.
    """
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd:
        pytesseract.pytesseract.tesseract_cmd = env_cmd
        return

    which = shutil.which("tesseract")
    if which:
        pytesseract.pytesseract.tesseract_cmd = which
        return

    if os.name == "nt":
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for c in candidates:
            if Path(c).exists():
                pytesseract.pytesseract.tesseract_cmd = c
                return


def extract_node_labels(bgr_img, nodes: list[Node]) -> dict[int, str]:
    """
    OCR each node bbox region and return {node_id: text}.

    Safe behavior:
    - If pytesseract isn't installed -> return {}
    - If tesseract isn't available -> return {}
    - If OCR errors on one node -> skip that node
    """
    try:
        import pytesseract
        from pytesseract import TesseractNotFoundError
    except ImportError:
        return {}

    _configure_tesseract_cmd(pytesseract)

    labels: dict[int, str] = {}

    for n in nodes:
        x, y, w, h = n.bbox
        roi = bgr_img[y : y + h, x : x + w]
        if roi.size == 0:
            continue

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
