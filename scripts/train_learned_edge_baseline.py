from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

from diagram2code.datasets.loader import load_dataset
from diagram2code.predictors.pairwise_features import extract_pair_features, feature_names


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


def _sample_edges(sample: Any) -> list[dict[str, Any]]:
    graph = _load_sample_graph_json(sample)
    return [dict(edge) for edge in graph.get("edges", [])]


def _sample_metadata(sample: Any) -> dict[str, Any]:
    graph = _load_sample_graph_json(sample)
    metadata = graph.get("metadata", {})
    return dict(metadata) if metadata is not None else {}


def _edge_endpoints(edge: dict[str, Any]) -> tuple[str, str]:
    if "from" in edge and "to" in edge:
        return str(edge["from"]), str(edge["to"])

    if "source" in edge and "target" in edge:
        return str(edge["source"]), str(edge["target"])

    if "src" in edge and "dst" in edge:
        return str(edge["src"]), str(edge["dst"])

    raise KeyError(f"Unsupported edge schema: {sorted(edge.keys())}")


def _resolve_split_samples(dataset: Any, split: str) -> list[Any]:
    samples_attr = getattr(dataset, "samples", None)

    if callable(samples_attr):
        try:
            return list(samples_attr(split))
        except TypeError:
            return [
                sample for sample in samples_attr() if _sample_field(sample, "split", None) == split
            ]

    if isinstance(samples_attr, dict):
        return list(samples_attr[split])

    if isinstance(samples_attr, (list, tuple)):
        return [sample for sample in samples_attr if _sample_field(sample, "split", None) == split]

    raise TypeError("Could not resolve samples for split from dataset object.")


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _sample_image_size(sample: Any) -> tuple[float, float]:
    metadata = _sample_metadata(sample)
    width = metadata.get("image_width")
    height = metadata.get("image_height")
    if width is not None and height is not None:
        return float(width), float(height)

    max_x = 1.0
    max_y = 1.0
    for node in _sample_nodes(sample):
        x, y, w, h = node["bbox"]
        max_x = max(max_x, float(x) + float(w))
        max_y = max(max_y, float(y) + float(h))
    return max_x, max_y


def _edge_set(sample: Any) -> set[tuple[str, str]]:
    return {(str(edge["from"]), str(edge["to"])) for edge in _sample_edges(sample)}


def _node_by_id(sample: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(node["id"]): node for node in sample["nodes"]}


def _negative_priority(
    source_node: dict[str, Any],
    target_node: dict[str, Any],
    image_width: float,
    image_height: float,
    nodes: list[dict[str, Any]],
) -> tuple[int, float]:
    feats = extract_pair_features(
        source_node=source_node,
        target_node=target_node,
        image_width=image_width,
        image_height=image_height,
        candidate_nodes=nodes,
    )
    names = feature_names()
    by_name = dict(zip(names, feats, strict=True))

    plausible = (
        by_name["same_row_like"] > 0.5
        or by_name["same_col_like"] > 0.5
        or by_name["is_nearest_right_neighbor"] > 0.5
        or by_name["is_nearest_down_neighbor"] > 0.5
    )
    priority = 0 if plausible else 1
    return priority, by_name["rank_by_distance_norm"]


def build_pairwise_training_examples(
    dataset_roots: list[Path],
    *,
    split: str,
    negative_ratio: int,
    seed: int,
) -> tuple[list[list[float]], list[int]]:
    rng = random.Random(seed)

    X: list[list[float]] = []
    y: list[int] = []

    for dataset_root in dataset_roots:
        dataset = load_dataset(dataset_root)
        samples = _resolve_split_samples(dataset, split)

        for sample in samples:
            nodes = _sample_nodes(sample)
            edges = _sample_edges(sample)

            width, height = _sample_image_size(sample)
            gold_edges = {_edge_endpoints(edge) for edge in edges}

            positives: list[tuple[dict[str, Any], dict[str, Any]]] = []
            negatives: list[tuple[dict[str, Any], dict[str, Any]]] = []

            for src in nodes:
                for dst in nodes:
                    if str(src["id"]) == str(dst["id"]):
                        continue
                    pair = (str(src["id"]), str(dst["id"]))
                    if pair in gold_edges:
                        positives.append((src, dst))
                    else:
                        negatives.append((src, dst))

            for src, dst in positives:
                X.append(
                    extract_pair_features(
                        source_node=src,
                        target_node=dst,
                        image_width=width,
                        image_height=height,
                        candidate_nodes=nodes,
                    )
                )
                y.append(1)

            if positives:
                negatives.sort(
                    key=lambda pair: _negative_priority(
                        source_node=pair[0],
                        target_node=pair[1],
                        image_width=width,
                        image_height=height,
                        nodes=nodes,
                    )
                )
                keep_n = min(len(negatives), negative_ratio * len(positives))
                selected = negatives[:keep_n]
            else:
                selected = []

            rng.shuffle(selected)

            for src, dst in selected:
                X.append(
                    extract_pair_features(
                        source_node=src,
                        target_node=dst,
                        image_width=width,
                        image_height=height,
                        candidate_nodes=nodes,
                    )
                )
                y.append(0)

    return X, y


def train_logistic_regression(
    X: list[list[float]],
    y: list[int],
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
    seed: int,
) -> tuple[list[float], float]:
    if not X:
        raise ValueError("No training examples were produced.")

    rng = random.Random(seed)
    n_features = len(X[0])
    weights = [0.0] * n_features
    bias = 0.0

    indices = list(range(len(X)))

    pos_count = sum(y)
    neg_count = len(y) - pos_count
    if pos_count == 0 or neg_count == 0:
        raise ValueError("Training requires both positive and negative examples.")

    # Balanced weighting
    pos_weight = len(y) / (2.0 * pos_count)
    neg_weight = len(y) / (2.0 * neg_count)

    for _epoch in range(epochs):
        rng.shuffle(indices)

        for idx in indices:
            feats = X[idx]
            label = y[idx]
            pred = _sigmoid(sum(w * x for w, x in zip(weights, feats, strict=True)) + bias)
            err = pred - float(label)
            sample_weight = pos_weight if label == 1 else neg_weight

            for j in range(n_features):
                grad = (err * feats[j] * sample_weight) + (l2 * weights[j])
                weights[j] -= learning_rate * grad

            bias -= learning_rate * err * sample_weight

    return weights, bias


def evaluate_training_fit(
    X: list[list[float]],
    y: list[int],
    weights: list[float],
    bias: float,
    *,
    threshold: float,
) -> dict[str, float]:
    tp = fp = tn = fn = 0

    for feats, label in zip(X, y, strict=True):
        prob = _sigmoid(sum(w * x for w, x in zip(weights, feats, strict=True)) + bias)
        pred = 1 if prob >= threshold else 0

        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        else:
            fn += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def export_model_artifact(
    *,
    out_path: Path,
    weights: list[float],
    bias: float,
    threshold: float,
    top_k: int,
) -> None:
    artifact = {
        "schema_version": 1,
        "model_type": "logistic_regression",
        "feature_names": feature_names(),
        "coef": weights,
        "intercept": bias,
        "threshold": threshold,
        "top_k": top_k,
    }
    out_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train learned edge baseline artifact.")
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        required=True,
        help="Dataset root path. Repeat --dataset for multiple datasets.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--negative-ratio", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default="src/diagram2code/predictors/learned_model.json",
        help="Output artifact path.",
    )
    args = parser.parse_args()

    dataset_roots = [Path(p) for p in args.datasets]
    out_path = Path(args.out)

    X, y = build_pairwise_training_examples(
        dataset_roots=dataset_roots,
        split=args.split,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
    )

    weights, bias = train_logistic_regression(
        X,
        y,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
    )

    metrics = evaluate_training_fit(
        X,
        y,
        weights,
        bias,
        threshold=args.threshold,
    )

    export_model_artifact(
        out_path=out_path,
        weights=weights,
        bias=bias,
        threshold=args.threshold,
        top_k=args.top_k,
    )

    label_counts = Counter(y)

    print(f"examples={len(y)}")
    print(f"positives={label_counts.get(1, 0)}")
    print(f"negatives={label_counts.get(0, 0)}")
    print(f"precision={metrics['precision']:.4f}")
    print(f"recall={metrics['recall']:.4f}")
    print(f"f1={metrics['f1']:.4f}")
    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"wrote={out_path}")


if __name__ == "__main__":
    main()
