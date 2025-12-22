from __future__ import annotations

from pathlib import Path
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


def _center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x, y, w, h = bbox
    return (x + w // 2, y + h // 2)


def detect_arrow_edges(
    binary_img: np.ndarray,
    nodes: List[Node],
    min_area: int = 80,
    max_area: int = 20000,
    debug_path: str | Path | None = None,
) -> List[Tuple[int, int]]:
    """
    Detect directed edges between nodes.
    Robust when arrows touch nodes by masking node rectangles out first.
    Returns list of (source_id, target_id).
    """
    # 1) Remove node rectangles from the binary image so arrows become separate components
    work = binary_img.copy()
    h, w = work.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    pad = 3
    for n in nodes:
        x, y, bw, bh = n.bbox
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w - 1, x + bw + pad)
        y1 = min(h - 1, y + bh + pad)
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)

    work[mask > 0] = 0

    # close small gaps
    kernel = np.ones((3, 3), np.uint8)
    work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    edges: List[Tuple[int, int]] = []
    debug_segments: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        pts = cnt.reshape(-1, 2)
        if pts.shape[0] < 5:
            continue

        xs = pts[:, 0]
        ys = pts[:, 1]
        dx = int(xs.max() - xs.min())
        dy = int(ys.max() - ys.min())

        # choose direction axis
        if dx >= dy:
            tail_pt = pts[np.argmin(xs)]
            head_pt = pts[np.argmax(xs)]
        else:
            tail_pt = pts[np.argmin(ys)]
            head_pt = pts[np.argmax(ys)]

        tail_id = _nearest_node_id(int(tail_pt[0]), int(tail_pt[1]), nodes)
        head_id = _nearest_node_id(int(head_pt[0]), int(head_pt[1]), nodes)

        if tail_id is None or head_id is None:
            continue
        if tail_id == head_id:
            continue

        edges.append((tail_id, head_id))
        debug_segments.append(((int(tail_pt[0]), int(tail_pt[1])), (int(head_pt[0]), int(head_pt[1]))))

    edges = sorted(set(edges))

    # Debug overlay
    if debug_path is not None:
        debug_path = Path(debug_path)
        debug_path.parent.mkdir(parents=True, exist_ok=True)

        vis = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)

        # nodes in green
        for n in nodes:
            x, y, bw, bh = n.bbox
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"Node {n.id}",
                (x, max(0, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # edges between centers in red
        id_to_node = {n.id: n for n in nodes}
        for a, b in edges:
            if a in id_to_node and b in id_to_node:
                p1 = _center(id_to_node[a].bbox)
                p2 = _center(id_to_node[b].bbox)
                cv2.arrowedLine(vis, p1, p2, (0, 0, 255), 2, tipLength=0.25)

        # raw tail/head points in blue
        for tail_pt, head_pt in debug_segments:
            cv2.circle(vis, tail_pt, 4, (255, 0, 0), -1)
            cv2.circle(vis, head_pt, 4, (255, 0, 0), -1)

        cv2.imwrite(str(debug_path), vis)

    return edges

