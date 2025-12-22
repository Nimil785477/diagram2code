from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

from diagram2code.schema import Node


def _point_to_bbox_dist2(px: int, py: int, bbox: Tuple[int, int, int, int]) -> int:
    x, y, w, h = bbox
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
    min_area: int = 80,
    max_area: int = 20000,
    debug_path: Optional[str | Path] = None,
) -> List[Tuple[int, int]]:
    """
    Detect directed edges between nodes.
    Works even if arrows touch node rectangles by masking nodes out first.
    Returns list of (source_id, target_id).
    """

    # 1) Remove node rectangles from the binary image so arrows become separate components
    work = binary_img.copy()
    h, w = work.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    pad = 3  # small padding helps if arrow touches node border
    for n in nodes:
        x, y, bw, bh = n.bbox
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w - 1, x + bw + pad)
        y1 = min(h - 1, y + bh + pad)
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)

    # set node pixels to black
    work[mask > 0] = 0

    # optional: close small gaps in arrow strokes
    kernel = np.ones((3, 3), np.uint8)
    work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 2) Find contours on arrows-only image
    contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    edges: List[Tuple[int, int]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        pts = cnt.reshape(-1, 2)
        if pts.shape[0] < 5:
            continue

        # 3) Determine main direction (horizontal vs vertical) using spread
        xs = pts[:, 0]
        ys = pts[:, 1]
        dx = int(xs.max() - xs.min())
        dy = int(ys.max() - ys.min())

        if dx >= dy:
            # horizontal-ish: tail is leftmost, head is rightmost
            tail_pt = pts[np.argmin(xs)]
            head_pt = pts[np.argmax(xs)]
        else:
            # vertical-ish: tail is topmost, head is bottommost
            tail_pt = pts[np.argmin(ys)]
            head_pt = pts[np.argmax(ys)]

        tail_id = _nearest_node_id(int(tail_pt[0]), int(tail_pt[1]), nodes)
        head_id = _nearest_node_id(int(head_pt[0]), int(head_pt[1]), nodes)

        if tail_id is None or head_id is None:
            continue
        if tail_id == head_id:
            continue

        edges.append((tail_id, head_id))

    edges = sorted(set(edges))

    # 4) Debug overlay
    if debug_path is not None:
        debug_path = Path(debug_path)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        dbg = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)

        # nodes
        for n in nodes:
            x, y, bw, bh = n.bbox
            cv2.rectangle(dbg, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(dbg, f"{n.id}", (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # edges as arrows between centers
        def center(bb):
            x, y, bw, bh = bb
            return (x + bw // 2, y + bh // 2)

        for a, b in edges:
            ca = center(nodes[a].bbox)
            cb = center(nodes[b].bbox)
            cv2.arrowedLine(dbg, ca, cb, (255, 0, 0), 2, tipLength=0.2)

        cv2.imwrite(str(debug_path), dbg)

    return edges
