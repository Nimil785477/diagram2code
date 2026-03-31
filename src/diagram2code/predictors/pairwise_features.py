from __future__ import annotations

import math
from typing import Any

EPS = 1e-9

FEATURE_NAMES = [
    "dx_norm",
    "dy_norm",
    "abs_dx_norm",
    "abs_dy_norm",
    "euclidean_norm",
    "manhattan_norm",
    "src_w_norm",
    "src_h_norm",
    "dst_w_norm",
    "dst_h_norm",
    "src_area_norm",
    "dst_area_norm",
    "log_w_ratio",
    "log_h_ratio",
    "log_area_ratio",
    "is_right_of",
    "is_left_of",
    "is_below",
    "is_above",
    "vertical_overlap_ratio",
    "horizontal_overlap_ratio",
    "same_row_like",
    "same_col_like",
    "rank_by_distance_norm",
    "rank_by_abs_dx_norm",
    "rank_by_abs_dy_norm",
    "is_nearest_right_neighbor",
    "is_nearest_down_neighbor",
]


def feature_names() -> list[str]:
    return FEATURE_NAMES.copy()


def _bbox(node: dict[str, Any]) -> tuple[float, float, float, float]:
    x, y, w, h = node["bbox"]
    return float(x), float(y), float(w), float(h)


def _center(node: dict[str, Any]) -> tuple[float, float]:
    x, y, w, h = _bbox(node)
    return x + (w / 2.0), y + (h / 2.0)


def _area(node: dict[str, Any]) -> float:
    _, _, w, h = _bbox(node)
    return max(w, 0.0) * max(h, 0.0)


def _vertical_overlap_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax, ay, aw, ah = _bbox(a)
    bx, by, bw, bh = _bbox(b)
    top = max(ay, by)
    bottom = min(ay + ah, by + bh)
    overlap = max(0.0, bottom - top)
    denom = max(min(ah, bh), EPS)
    return overlap / denom


def _horizontal_overlap_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax, ay, aw, ah = _bbox(a)
    bx, by, bw, bh = _bbox(b)
    left = max(ax, bx)
    right = min(ax + aw, bx + bw)
    overlap = max(0.0, right - left)
    denom = max(min(aw, bw), EPS)
    return overlap / denom


def _safe_log_ratio(num: float, den: float) -> float:
    return math.log((num + EPS) / (den + EPS))


def _pair_distance(src: dict[str, Any], dst: dict[str, Any]) -> tuple[float, float, float, float]:
    src_cx, src_cy = _center(src)
    dst_cx, dst_cy = _center(dst)
    dx = dst_cx - src_cx
    dy = dst_cy - src_cy
    euclidean = math.hypot(dx, dy)
    manhattan = abs(dx) + abs(dy)
    return dx, dy, euclidean, manhattan


def _rank_candidates(
    source_node: dict[str, Any],
    candidate_nodes: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    source_id = source_node["id"]
    scored: list[tuple[str, float, float, float, float, float]] = []

    src_cx, src_cy = _center(source_node)

    for node in candidate_nodes:
        if node["id"] == source_id:
            continue
        dst_cx, dst_cy = _center(node)
        dx = dst_cx - src_cx
        dy = dst_cy - src_cy
        scored.append(
            (
                str(node["id"]),
                math.hypot(dx, dy),
                abs(dx),
                abs(dy),
                dx,
                dy,
            )
        )

    by_dist = {
        node_id: rank
        for rank, (node_id, *_rest) in enumerate(sorted(scored, key=lambda x: x[1]), start=1)
    }
    by_abs_dx = {
        node_id: rank
        for rank, (node_id, *_rest) in enumerate(sorted(scored, key=lambda x: x[2]), start=1)
    }
    by_abs_dy = {
        node_id: rank
        for rank, (node_id, *_rest) in enumerate(sorted(scored, key=lambda x: x[3]), start=1)
    }

    right_candidates = [row for row in scored if row[4] > 0]
    down_candidates = [row for row in scored if row[5] > 0]

    nearest_right_id = min(right_candidates, key=lambda x: x[1])[0] if right_candidates else None
    nearest_down_id = min(down_candidates, key=lambda x: x[1])[0] if down_candidates else None

    return {
        "by_dist": by_dist,
        "by_abs_dx": by_abs_dx,
        "by_abs_dy": by_abs_dy,
        "nearest_right": {"id": nearest_right_id},
        "nearest_down": {"id": nearest_down_id},
        "num_candidates": {"value": max(len(scored), 1)},
    }


def extract_pair_features(
    source_node: dict[str, Any],
    target_node: dict[str, Any],
    image_width: float,
    image_height: float,
    candidate_nodes: list[dict[str, Any]],
) -> list[float]:
    if source_node["id"] == target_node["id"]:
        raise ValueError("source_node and target_node must be different")

    image_width = max(float(image_width), 1.0)
    image_height = max(float(image_height), 1.0)
    image_diag = math.hypot(image_width, image_height)

    dx, dy, euclidean, manhattan = _pair_distance(source_node, target_node)

    _, _, src_w, src_h = _bbox(source_node)
    _, _, dst_w, dst_h = _bbox(target_node)
    src_area = _area(source_node)
    dst_area = _area(target_node)

    vertical_overlap_ratio = _vertical_overlap_ratio(source_node, target_node)
    horizontal_overlap_ratio = _horizontal_overlap_ratio(source_node, target_node)

    row_scale = max(src_h, dst_h, 1.0)
    col_scale = max(src_w, dst_w, 1.0)

    same_row_like = 1.0 if (abs(dy) / row_scale) < 0.75 else 0.0
    same_col_like = 1.0 if (abs(dx) / col_scale) < 0.75 else 0.0

    ranks = _rank_candidates(source_node, candidate_nodes)
    target_id = str(target_node["id"])
    num_candidates = float(ranks["num_candidates"]["value"])

    rank_by_distance_norm = ranks["by_dist"].get(target_id, num_candidates) / num_candidates
    rank_by_abs_dx_norm = ranks["by_abs_dx"].get(target_id, num_candidates) / num_candidates
    rank_by_abs_dy_norm = ranks["by_abs_dy"].get(target_id, num_candidates) / num_candidates

    is_nearest_right_neighbor = 1.0 if ranks["nearest_right"]["id"] == target_id else 0.0
    is_nearest_down_neighbor = 1.0 if ranks["nearest_down"]["id"] == target_id else 0.0

    features = [
        dx / image_width,
        dy / image_height,
        abs(dx) / image_width,
        abs(dy) / image_height,
        euclidean / image_diag,
        manhattan / (image_width + image_height),
        src_w / image_width,
        src_h / image_height,
        dst_w / image_width,
        dst_h / image_height,
        src_area / max(image_width * image_height, 1.0),
        dst_area / max(image_width * image_height, 1.0),
        _safe_log_ratio(dst_w, src_w),
        _safe_log_ratio(dst_h, src_h),
        _safe_log_ratio(dst_area, src_area),
        1.0 if dx > 0 else 0.0,
        1.0 if dx < 0 else 0.0,
        1.0 if dy > 0 else 0.0,
        1.0 if dy < 0 else 0.0,
        vertical_overlap_ratio,
        horizontal_overlap_ratio,
        same_row_like,
        same_col_like,
        rank_by_distance_norm,
        rank_by_abs_dx_norm,
        rank_by_abs_dy_norm,
        is_nearest_right_neighbor,
        is_nearest_down_neighbor,
    ]

    if len(features) != len(FEATURE_NAMES):
        raise AssertionError("feature vector length does not match FEATURE_NAMES")

    for value in features:
        if not math.isfinite(value):
            raise ValueError(f"non-finite feature value detected: {value}")

    return features
