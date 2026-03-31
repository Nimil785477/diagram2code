from __future__ import annotations

from pathlib import Path
from typing import Any

from .learned_model_artifact import LearnedEdgeScorer, LearnedModelArtifact
from .pairwise_features import extract_pair_features, feature_names


def _dedupe_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []

    for edge in edges:
        src = str(edge["from"])
        dst = str(edge["to"])
        key = (src, dst)
        if key in seen:
            continue
        seen.add(key)
        out.append({"from": src, "to": dst})

    return out


def _sample_image_size(sample: dict[str, Any]) -> tuple[float, float]:
    metadata = sample.get("metadata", {})
    width = metadata.get("image_width")
    height = metadata.get("image_height")

    if width is not None and height is not None:
        return float(width), float(height)

    # fallback: infer from node extents
    nodes = sample["nodes"]
    max_x = 0.0
    max_y = 0.0
    for node in nodes:
        x, y, w, h = node["bbox"]
        max_x = max(max_x, float(x) + float(w))
        max_y = max(max_y, float(y) + float(h))

    return max(max_x, 1.0), max(max_y, 1.0)


class LearnedPredictor:
    """
    Lightweight learned edge baseline.

    Expected input sample shape:
    {
        "nodes": [...],
        "metadata": {...},   # optional image_width / image_height
        ...
    }

    Expected output:
    {
        "nodes": [...],
        "edges": [{"from": "...", "to": "..."}, ...]
    }
    """

    name = "learned"
    description = "Learned pairwise edge baseline using geometric features"

    def __init__(self, model_path: str | Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).with_name("learned_model.json")

        artifact = LearnedModelArtifact.from_path(model_path)
        self._scorer = LearnedEdgeScorer(artifact)

        expected_names = feature_names()
        if artifact.feature_names != expected_names:
            raise ValueError("learned model feature_names do not match runtime feature_names")

    def predict(self, sample: dict[str, Any]) -> dict[str, Any]:
        nodes = [dict(node) for node in sample["nodes"]]
        image_width, image_height = _sample_image_size(sample)

        scored_edges: list[tuple[float, str, str]] = []

        for src in nodes:
            local_scores: list[tuple[float, str, str]] = []

            for dst in nodes:
                if str(src["id"]) == str(dst["id"]):
                    continue

                feats = extract_pair_features(
                    source_node=src,
                    target_node=dst,
                    image_width=image_width,
                    image_height=image_height,
                    candidate_nodes=nodes,
                )
                score = self._scorer.score(feats)
                local_scores.append((score, str(src["id"]), str(dst["id"])))

            local_scores.sort(key=lambda row: row[0], reverse=True)

            kept = 0
            for score, src_id, dst_id in local_scores:
                if score < self._scorer.threshold:
                    continue
                if kept >= self._scorer.top_k:
                    break
                scored_edges.append((score, src_id, dst_id))
                kept += 1

        edges = _dedupe_edges(
            [{"from": src_id, "to": dst_id} for _score, src_id, dst_id in scored_edges]
        )

        return {
            "nodes": nodes,
            "edges": edges,
        }
