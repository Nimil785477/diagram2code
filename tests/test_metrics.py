from diagram2code.benchmark.metrics import (
    compute_direction_accuracy,
    compute_edge_metrics,
    compute_exact_match,
    compute_node_metrics,
)


def test_node_prf1_simple():
    gt_nodes = [{"id": "g1", "bbox": [0, 0, 10, 10]}, {"id": "g2", "bbox": [20, 0, 10, 10]}]
    pred_nodes = [{"id": "p1", "bbox": [0, 0, 10, 10]}, {"id": "p2", "bbox": [20, 0, 10, 10]}]
    pred_to_gt = {"p1": "g1"}  # 1 match

    m = compute_node_metrics(gt_nodes=gt_nodes, pred_nodes=pred_nodes, pred_to_gt=pred_to_gt)
    assert m.precision == 1 / 2
    assert m.recall == 1 / 2
    assert m.f1 == 0.5


def test_edge_prf1_simple():
    gt_edges = [{"from": "g1", "to": "g2"}]
    projected = {("g1", "g2"), ("g2", "g1")}  # one correct, one extra

    m = compute_edge_metrics(gt_edges=gt_edges, projected_pred_edges_gt_space=projected)
    assert m.precision == 1 / 2
    assert m.recall == 1 / 1
    assert m.f1 == 2 * (0.5 * 1.0) / (0.5 + 1.0)


def test_direction_accuracy_counts_only_edges_that_map_to_gt_connection():
    gt_edges = [{"from": "g1", "to": "g2"}]
    projected = {
        ("g1", "g2"),  # correct direction
        ("g2", "g1"),  # wrong direction but corresponds to GT connection
        ("g9", "g10"),  # unrelated; should not count in denom
    }

    acc = compute_direction_accuracy(gt_edges=gt_edges, projected_pred_edges_gt_space=projected)
    assert acc == 1 / 2


def test_exact_match_true():
    gt_nodes = [{"id": "g1", "bbox": [0, 0, 10, 10]}, {"id": "g2", "bbox": [20, 0, 10, 10]}]
    gt_edges = [{"from": "g1", "to": "g2"}]

    pred_nodes = [{"id": "p1", "bbox": [0, 0, 10, 10]}, {"id": "p2", "bbox": [20, 0, 10, 10]}]
    pred_to_gt = {"p1": "g1", "p2": "g2"}
    projected = {("g1", "g2")}

    assert compute_exact_match(
        gt_nodes=gt_nodes,
        gt_edges=gt_edges,
        pred_nodes=pred_nodes,
        projected_pred_edges_gt_space=projected,
        pred_to_gt=pred_to_gt,
    )


def test_exact_match_false_if_edges_differ():
    gt_nodes = [{"id": "g1", "bbox": [0, 0, 10, 10]}, {"id": "g2", "bbox": [20, 0, 10, 10]}]
    gt_edges = [{"from": "g1", "to": "g2"}]

    pred_nodes = [{"id": "p1", "bbox": [0, 0, 10, 10]}, {"id": "p2", "bbox": [20, 0, 10, 10]}]
    pred_to_gt = {"p1": "g1", "p2": "g2"}
    projected = {("g2", "g1")}  # reversed

    assert not compute_exact_match(
        gt_nodes=gt_nodes,
        gt_edges=gt_edges,
        pred_nodes=pred_nodes,
        projected_pred_edges_gt_space=projected,
        pred_to_gt=pred_to_gt,
    )
