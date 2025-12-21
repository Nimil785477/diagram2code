from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from diagram2code.schema import Node


def _point_to_bbox_dist2(px: int, py: int, bbox: Tuple[int, int, int, int]) -> int:
    x, y, w, h = bbox
    # clamp point to bbox
    cx = min(max(px, x), x + w)
    cy = min(max(py, y), y + h)
    dx = px - cx
    dy = py - cy
    return dx * dx + dy * dy


def _nearest_node_id(px: int, py: int, nodes: List[Node]) -> int | None:
    if not nodes:
        return None
    best = min((_point_to_bbox_dist2(px, py, n.bbox), n.id) for n in nodes)
    return best[1]


def detect_arrow_edges(
    binary_img: np.ndarray,
    nodes: List[Node],
    min_area: int = 150,
    max_area: int = 5000,
) -> List[Tuple[int, int]]:
    """
    Detect directed edges between nodes using arrow-like contours.
    Assumes arrows are smaller than node rectangles.
    Returns list of (source_id, target_id).
    """
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    edges: List[Tuple[int, int]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # ignore likely node rectangles (big blocks)
        if w > 60 and h > 60:
            continue

        pts = cnt.reshape(-1, 2)
        # tail/head by extreme x (works for your fixture's horizontal arrow)
        left = pts[np.argmin(pts[:, 0])]
        right = pts[np.argmax(pts[:, 0])]

        tail_id = _nearest_node_id(int(left[0]), int(left[1]), nodes)
        head_id = _nearest_node_id(int(right[0]), int(right[1]), nodes)

        if tail_id is None or head_id is None:
            continue
        if tail_id == head_id:
            continue

        edges.append((tail_id, head_id))

    # dedupe stable
    edges = sorted(set(edges))
    return edges
