from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PRF1:
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class BenchmarkMetrics:
    node: PRF1
    edge: PRF1
    direction_accuracy: float
    exact_match: bool
    runtime_s: float | None = None


def _safe_prf1(tp: int, pred_n: int, gt_n: int) -> PRF1:
    precision = tp / pred_n if pred_n > 0 else 0.0
    recall = tp / gt_n if gt_n > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return PRF1(precision=precision, recall=recall, f1=f1)


def compute_node_metrics(
    *,
    gt_nodes: list[dict],
    pred_nodes: list[dict],
    pred_to_gt: dict[str, str],
) -> PRF1:
    """
    Node TP = number of matched predicted nodes (one-to-one mapping).
    """
    tp = len(pred_to_gt)
    return _safe_prf1(tp=tp, pred_n=len(pred_nodes), gt_n=len(gt_nodes))


def _edge_set(edges: list[dict]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for e in edges:
        out.add((str(e["from"]), str(e["to"])))
    return out


def compute_edge_metrics(
    *,
    gt_edges: list[dict],
    projected_pred_edges_gt_space: set[tuple[str, str]],
) -> PRF1:
    """
    Edge TP = | projected_pred_edges ∩ gt_edges |.
    """
    gt = _edge_set(gt_edges)
    tp = len(projected_pred_edges_gt_space & gt)
    return _safe_prf1(tp=tp, pred_n=len(projected_pred_edges_gt_space), gt_n=len(gt))


def compute_direction_accuracy(
    *,
    gt_edges: list[dict],
    projected_pred_edges_gt_space: set[tuple[str, str]],
) -> float:
    """
    Direction accuracy among predicted edges that correspond to a GT connection
    (in either direction).

    Denominator = count of predicted edges where either (a,b) or (b,a) exists in GT.
    Numerator = count of those predicted edges where (a,b) exists in GT exactly.
    """
    gt = _edge_set(gt_edges)
    denom = 0
    correct = 0

    for a, b in projected_pred_edges_gt_space:
        if (a, b) in gt:
            denom += 1
            correct += 1
        elif (b, a) in gt:
            denom += 1

    return correct / denom if denom > 0 else 0.0


def compute_exact_match(
    *,
    gt_nodes: list[dict],
    gt_edges: list[dict],
    pred_nodes: list[dict],
    projected_pred_edges_gt_space: set[tuple[str, str]],
    pred_to_gt: dict[str, str],
) -> bool:
    """
    Exact match requires:
      - all GT nodes matched and counts equal
      - edge set matches exactly in GT-id space
    """
    gt_node_ids = {str(n["id"]) for n in gt_nodes}
    pred_node_ids = {str(n["id"]) for n in pred_nodes}

    # Node exact: bijection by id count and full coverage of GT
    node_exact = (
        len(pred_node_ids) == len(gt_node_ids)
        and len(pred_to_gt) == len(gt_node_ids)
        and set(pred_to_gt.values()) == gt_node_ids
    )

    gt_edge_set = _edge_set(gt_edges)
    edge_exact = projected_pred_edges_gt_space == gt_edge_set

    return node_exact and edge_exact


def compute_metrics(
    *,
    gt_nodes: list[dict],
    gt_edges: list[dict],
    pred_nodes: list[dict],
    pred_edges_projected_gt_space: set[tuple[str, str]],
    pred_to_gt: dict[str, str],
    runtime_s: float | None = None,
) -> BenchmarkMetrics:
    node = compute_node_metrics(gt_nodes=gt_nodes, pred_nodes=pred_nodes, pred_to_gt=pred_to_gt)
    edge = compute_edge_metrics(
        gt_edges=gt_edges, projected_pred_edges_gt_space=pred_edges_projected_gt_space
    )
    direction_accuracy = compute_direction_accuracy(
        gt_edges=gt_edges,
        projected_pred_edges_gt_space=pred_edges_projected_gt_space,
    )
    exact_match = compute_exact_match(
        gt_nodes=gt_nodes,
        gt_edges=gt_edges,
        pred_nodes=pred_nodes,
        projected_pred_edges_gt_space=pred_edges_projected_gt_space,
        pred_to_gt=pred_to_gt,
    )
    return BenchmarkMetrics(
        node=node,
        edge=edge,
        direction_accuracy=direction_accuracy,
        exact_match=exact_match,
        runtime_s=runtime_s,
    )
