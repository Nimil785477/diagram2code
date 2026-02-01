from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from diagram2code.benchmark.matching import (
    match_nodes_center_distance,
    project_pred_edges_to_gt,
)
from diagram2code.benchmark.metrics import (
    PRF1,
    BenchmarkMetrics,
    compute_metrics,
)
from diagram2code.benchmark.predictor import PredGraph, Predictor


@dataclass(frozen=True)
class SampleResult:
    image_path: Path
    metrics: BenchmarkMetrics


@dataclass(frozen=True)
class AggregateResult:
    node: PRF1
    edge: PRF1
    direction_accuracy: float
    exact_match_rate: float
    runtime_mean_s: float | None


@dataclass(frozen=True)
class BenchmarkResult:
    samples: list[SampleResult]
    aggregate: AggregateResult


def _node_to_dict(n) -> dict:
    if isinstance(n, dict):
        return n
    return {"id": n.id, "bbox": list(n.bbox)}


def _edge_to_dict(e) -> dict:
    if isinstance(e, dict):
        return e
    if hasattr(e, "from_id") and hasattr(e, "to_id"):
        return {"from": e.from_id, "to": e.to_id}
    # If your edge dataclass uses different names, adjust here when traceback tells us.
    return {"from": e.from_, "to": e.to}


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _mean_optional(xs: list[float | None]) -> float | None:
    ys = [x for x in xs if x is not None]
    return _mean(ys) if ys else None


def _aggregate(samples: list[SampleResult]) -> AggregateResult:
    node_p = _mean([s.metrics.node.precision for s in samples])
    node_r = _mean([s.metrics.node.recall for s in samples])
    node_f = _mean([s.metrics.node.f1 for s in samples])

    edge_p = _mean([s.metrics.edge.precision for s in samples])
    edge_r = _mean([s.metrics.edge.recall for s in samples])
    edge_f = _mean([s.metrics.edge.f1 for s in samples])

    dir_acc = _mean([s.metrics.direction_accuracy for s in samples])
    exact_rate = _mean([1.0 if s.metrics.exact_match else 0.0 for s in samples])
    rt_mean = _mean_optional([s.metrics.runtime_s for s in samples])

    return AggregateResult(
        node=PRF1(node_p, node_r, node_f),
        edge=PRF1(edge_p, edge_r, edge_f),
        direction_accuracy=dir_acc,
        exact_match_rate=exact_rate,
        runtime_mean_s=rt_mean,
    )


def run_benchmark(
    *,
    dataset_path: Path,
    predictor: Predictor,
    alpha: float,
) -> BenchmarkResult:
    """
    Library-only benchmark runner (no CLI).

    Pipeline per sample:
    - Load GT graph from dataset (Step 1)
    - Predict nodes/edges for image
    - Match pred->gt nodes with alpha (Step 2)
    - Project pred edges into GT-id space (Step 2)
    - Compute metrics (Step 3)
    - Aggregate over dataset
    """
    from diagram2code.benchmark.dataset import load_dataset

    dataset = load_dataset(dataset_path)
    items = dataset if isinstance(dataset, list) else dataset.items  # Dataset(root, name, items)

    per_sample: list[SampleResult] = []
    for item in items:
        # DatasetItem fields: image_path, graph_path, gt
        image_path = item.image_path
        gt_graph = item.gt

        gt_nodes_raw = gt_graph["nodes"] if isinstance(gt_graph, dict) else gt_graph.nodes
        gt_edges_raw = gt_graph["edges"] if isinstance(gt_graph, dict) else gt_graph.edges

        gt_nodes = [_node_to_dict(n) for n in gt_nodes_raw]
        gt_edges = [_edge_to_dict(e) for e in gt_edges_raw]

        pred: PredGraph = predictor(image_path)

        pred_to_gt = match_nodes_center_distance(
            gt_nodes=gt_nodes,
            pred_nodes=pred.nodes,
            alpha=alpha,
        )
        projected_edges = project_pred_edges_to_gt(pred.edges, pred_to_gt)

        m = compute_metrics(
            gt_nodes=gt_nodes,
            gt_edges=gt_edges,
            pred_nodes=pred.nodes,
            pred_edges_projected_gt_space=projected_edges,
            pred_to_gt=pred_to_gt,
            runtime_s=None,
        )

        per_sample.append(SampleResult(image_path=image_path, metrics=m))

    return BenchmarkResult(samples=per_sample, aggregate=_aggregate(per_sample))
