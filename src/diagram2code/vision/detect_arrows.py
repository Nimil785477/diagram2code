from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

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


def _center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x, y, w, h = bbox
    return (x + w // 2, y + h // 2)


def _angle_at(p_prev: np.ndarray, p: np.ndarray, p_next: np.ndarray) -> float:
    """Angle (radians) at p. Smaller = sharper."""
    v1 = (p_prev - p).astype(np.float32)
    v2 = (p_next - p).astype(np.float32)
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return float("inf")
    v1 /= n1
    v2 /= n2
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return float(np.arccos(dot))


def _pca_axis(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA axis for points:
      - mean (2,)
      - unit axis v (2,)
      - projections proj (N,)
      - endpoints (pt_min, pt_max) along axis projection
    """
    pts_f = pts.astype(np.float32)
    mean = pts_f.mean(axis=0)
    centered = pts_f - mean

    cov = np.cov(centered.T)
    if cov.shape != (2, 2) or not np.isfinite(cov).all():
        v = np.array([1.0, 0.0], dtype=np.float32)
    else:
        eigvals, eigvecs = np.linalg.eigh(cov)
        v = eigvecs[:, int(np.argmax(eigvals))].astype(np.float32)

    v_norm = float(np.linalg.norm(v))
    if v_norm < 1e-6:
        v = np.array([1.0, 0.0], dtype=np.float32)
    else:
        v /= v_norm

    proj = centered @ v
    pt_min = pts[int(np.argmin(proj))]
    pt_max = pts[int(np.argmax(proj))]
    return mean, v, proj, pt_min, pt_max


def _tail_head_from_hull_angles(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Determine arrow direction by:
      1) PCA to get axis and min/max ends.
      2) Convex hull angles: find most acute corner near each end.
      3) For masked arrow components in this project, the *tail* end often becomes the
         sharpest due to truncation. So we invert the naive rule:
            head end = side whose local min-angle is *larger*.
    """
    mean, v, _proj, pt_min, pt_max = _pca_axis(pts)

    hull = cv2.convexHull(pts.reshape(-1, 1, 2), returnPoints=True).reshape(-1, 2)
    if hull.shape[0] < 5:
        # Not enough structure -> fall back to PCA orientation
        return pt_min, pt_max

    hull_f = hull.astype(np.float32)
    centered_h = hull_f - mean.astype(np.float32)
    hproj = centered_h @ v
    hmin = float(hproj.min())
    hmax = float(hproj.max())
    span = hmax - hmin
    if span <= 1e-6:
        return pt_min, pt_max

    band = 0.15 * span  # 15% near each end

    # Compute angles for hull vertices
    n = hull.shape[0]
    angles = np.empty((n,), dtype=np.float32)
    for i in range(n):
        angles[i] = _angle_at(hull[(i - 1) % n], hull[i], hull[(i + 1) % n])

    # Candidates near min and max ends
    min_mask = hproj <= (hmin + band)
    max_mask = hproj >= (hmax - band)

    # If too few points, widen band
    if int(min_mask.sum()) < 2:
        min_mask = hproj <= (hmin + 0.25 * span)
    if int(max_mask.sum()) < 2:
        max_mask = hproj >= (hmax - 0.25 * span)

    min_angles = angles[min_mask]
    max_angles = angles[max_mask]
    if min_angles.size == 0 or max_angles.size == 0:
        return pt_min, pt_max

    min_best = float(min_angles.min())
    max_best = float(max_angles.min())

    # IMPORTANT: invert decision (see docstring)
    # head end is the side with the *larger* min-angle
    if max_best > min_best:
        head_side = "max"
    elif min_best > max_best:
        head_side = "min"
    else:
        head_side = "max"

    if head_side == "max":
        idxs = np.where(max_mask)[0]
    else:
        idxs = np.where(min_mask)[0]

    side_angles = angles[idxs]
    head_idx = int(idxs[int(np.argmin(side_angles))])
    head_pt = hull[head_idx]

    # tail: farthest hull point from head (robust)
    d2 = np.sum((hull.astype(np.int32) - head_pt.astype(np.int32)) ** 2, axis=1)
    tail_pt = hull[int(np.argmax(d2))]

    return tail_pt, head_pt


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
        if pts.shape[0] < 10:
            continue

        tail_pt, head_pt = _tail_head_from_hull_angles(pts)

        tail_id = _nearest_node_id(int(tail_pt[0]), int(tail_pt[1]), nodes)
        head_id = _nearest_node_id(int(head_pt[0]), int(head_pt[1]), nodes)

        if tail_id is None or head_id is None:
            continue
        if tail_id == head_id:
            continue

        edges.append((tail_id, head_id))
        debug_segments.append(((int(tail_pt[0]), int(tail_pt[1])), (int(head_pt[0]), int(head_pt[1]))))

    edges = sorted(set(edges))

    if debug_path is not None:
        debug_path = Path(debug_path)
        debug_path.parent.mkdir(parents=True, exist_ok=True)

        vis = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)

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

        id_to_node = {n.id: n for n in nodes}
        for a, b in edges:
            if a in id_to_node and b in id_to_node:
                p1 = _center(id_to_node[a].bbox)
                p2 = _center(id_to_node[b].bbox)
                cv2.arrowedLine(vis, p1, p2, (0, 0, 255), 2, tipLength=0.25)

        for tpt, hpt in debug_segments:
            cv2.circle(vis, tpt, 4, (255, 0, 0), -1)
            cv2.circle(vis, hpt, 4, (255, 0, 0), -1)

        cv2.imwrite(str(debug_path), vis)

    return edges
