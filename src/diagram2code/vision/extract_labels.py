from __future__ import annotations
from typing import Dict, List
import cv2
import pytesseract

from diagram2code.schema import Node
from diagram2code.labels import to_valid_identifier


def extract_node_labels(
    bgr_img,
    nodes: List[Node],
) -> Dict[int, str]:
    labels: Dict[int, str] = {}

    for n in nodes:
        x, y, w, h = n.bbox
        crop = bgr_img[y:y+h, x:x+w]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        text = pytesseract.image_to_string(
            gray,
            config="--psm 6"
        ).strip()

        if text:
            labels[n.id] = to_valid_identifier(text, fallback=f"node_{n.id}")

    return labels
