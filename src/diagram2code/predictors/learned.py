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


def _sample_field(sample: Any, name: str, default: Any = None) -> Any:
    if isinstance(sample, dict):
        return sample.get(name, default)
    return getattr(sample, name, default)


def _load_sample_graph_json(sample: Any) -> dict[str, Any]:
    if isinstance(sample, dict):
        if "graph" in sample and isinstance(sample["graph"], dict):
            return dict(sample["graph"])
        return dict(sample)

    load_graph_json = getattr(sample, "load_graph_json", None)
    if callable(load_graph_json):
        return dict(load_graph_json())

    graph = getattr(sample, "graph", None)
    if isinstance(graph, dict):
        return dict(graph)

    raise TypeError("Could not load graph JSON from sample")


def _sample_nodes(sample: Any) -> list[dict[str, Any]]:
    graph = _load_sample_graph_json(sample)
    return [dict(node) for node in graph.get("nodes", [])]


def _sample_metadata(sample: Any) -> dict[str, Any]:
    graph = _load_sample_graph_json(sample)
    metadata = graph.get("metadata", {})
    return dict(metadata) if metadata is not None else {}


def _sample_image_size(sample: Any) -> tuple[float, float]:
    metadata = _sample_metadata(sample)
    width = metadata.get("image_width")
    height = metadata.get("image_height")

    if width is not None and height is not None:
        return float(width), float(height)

    nodes = _sample_nodes(sample)
    max_x = 0.0
    max_y = 0.0
    for node in nodes:
        x, y, w, h = node["bbox"]
        max_x = max(max_x, float(x) + float(w))
        max_y = max(max_y, float(y) + float(h))

    return max(max_x, 1.0), max(max_y, 1.0)


class LearnedPredictor:
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

    def predict(self, sample: Any) -> dict[str, Any]:
        nodes = _sample_nodes(sample)
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
