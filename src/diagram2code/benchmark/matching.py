from __future__ import annotations

from collections.abc import Iterable
from math import hypot


class MatchingError(ValueError):
    """Raised when matching inputs are invalid."""


def _bbox_center_and_diag(bbox: list[float] | tuple[float, float, float, float]):
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0
    diag = hypot(w, h)
    return (cx, cy), diag


def match_nodes_center_distance(
    gt_nodes: list[dict],
    pred_nodes: list[dict],
    *,
    alpha: float,
) -> dict[str, str]:
    """
    Match predicted nodes to GT nodes using center-distance rule.

    A pred node matches a GT node iff:
        dist(center_pred, center_gt) <= alpha * diag(gt_bbox)

    Returns:
        dict[pred_id -> gt_id]
    """
    if alpha <= 0:
        raise MatchingError("alpha must be > 0")

    gt_info: dict[str, tuple[tuple[float, float], float]] = {}
    for n in gt_nodes:
        if "id" not in n or "bbox" not in n:
            raise MatchingError("GT nodes must have id and bbox")
        gt_info[str(n["id"])] = _bbox_center_and_diag(n["bbox"])

    pred_centers: dict[str, tuple[float, float]] = {}
    for n in pred_nodes:
        if "id" not in n or "bbox" not in n:
            raise MatchingError("Pred nodes must have id and bbox")
        center, _ = _bbox_center_and_diag(n["bbox"])
        pred_centers[str(n["id"])] = center

    # candidate matches: (distance, pred_id, gt_id)
    candidates: list[tuple[float, str, str]] = []
    for pid, (px, py) in pred_centers.items():
        for gid, ((gx, gy), gdiag) in gt_info.items():
            d = hypot(px - gx, py - gy)
            if d <= alpha * gdiag:
                candidates.append((d, pid, gid))

    candidates.sort(key=lambda t: (t[0], t[1], t[2]))

    pred_to_gt: dict[str, str] = {}
    used_preds: set[str] = set()
    used_gts: set[str] = set()

    for _, pid, gid in candidates:
        if pid in used_preds or gid in used_gts:
            continue
        pred_to_gt[pid] = gid
        used_preds.add(pid)
        used_gts.add(gid)

    return pred_to_gt


def project_pred_edges_to_gt(
    pred_edges: Iterable[dict],
    pred_to_gt: dict[str, str],
) -> set[tuple[str, str]]:
    """
    Convert predicted edges into GT-id space.

    Unmatched endpoints are dropped.
    """
    out: set[tuple[str, str]] = set()

    for e in pred_edges:
        if "from" not in e or "to" not in e:
            raise MatchingError("Edges must have 'from' and 'to'")
        pf = str(e["from"])
        pt = str(e["to"])

        gf = pred_to_gt.get(pf)
        gt = pred_to_gt.get(pt)

        if gf is None or gt is None:
            continue

        out.add((gf, gt))

    return out
