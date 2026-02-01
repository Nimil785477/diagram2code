from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from diagram2code.schema import Node


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    a2x, a2y = ax + aw, ay + ah
    b2x, b2y = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(a2x, b2x), min(a2y, b2y)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _dedupe_bboxes(bboxes: list[tuple[int, int, int, int]], iou_thresh: float = 0.7):
    kept: list[tuple[int, int, int, int]] = []
    for bb in sorted(bboxes, key=lambda x: x[2] * x[3], reverse=True):
        if all(_iou(bb, k) < iou_thresh for k in kept):
            kept.append(bb)
    return kept


def detect_rectangles(
    binary_img: np.ndarray,
    min_area: int = 800,
    debug_path: str | Path | None = None,
) -> list[Node]:
    """
    Detect rectangular nodes in a preprocessed (binary) image.
    Expects white shapes on black background (THRESH_BINARY_INV style).
    """
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes: list[tuple[int, int, int, int]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)

        # avoid long thin lines being counted as rectangles
        if w < 20 or h < 20:
            continue

        aspect = w / float(h)
        if aspect < 0.4 or aspect > 2.5:
            continue

        bboxes.append((x, y, w, h))

    bboxes = _dedupe_bboxes(bboxes)

    # stable ordering: left-to-right then top-to-bottom
    bboxes = sorted(bboxes, key=lambda bb: (bb[0], bb[1]))

    nodes = [Node(i, bb) for i, bb in enumerate(bboxes)]

    if debug_path is not None:
        dbg = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        dbg = draw_nodes_on_image(dbg, nodes)
        cv2.imwrite(str(Path(debug_path)), dbg)

    return nodes


def draw_nodes_on_image(bgr_img: np.ndarray, nodes: list[Node]) -> np.ndarray:
    out = bgr_img.copy()
    for n in nodes:
        x, y, w, h = n.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"Node {n.id}",
            (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return out
