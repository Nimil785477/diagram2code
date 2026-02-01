import pytest

from diagram2code.benchmark.matching import (
    MatchingError,
    match_nodes_center_distance,
    project_pred_edges_to_gt,
)


def test_basic_one_to_one_matching():
    gt_nodes = [
        {"id": "g1", "bbox": [0, 0, 100, 100]},
        {"id": "g2", "bbox": [200, 0, 100, 100]},
    ]
    pred_nodes = [
        {"id": "p1", "bbox": [10, 10, 80, 80]},
        {"id": "p2", "bbox": [210, 10, 80, 80]},
    ]

    m = match_nodes_center_distance(gt_nodes, pred_nodes, alpha=0.3)
    assert m == {"p1": "g1", "p2": "g2"}


def test_distance_threshold_blocks_match():
    gt_nodes = [{"id": "g1", "bbox": [0, 0, 100, 100]}]
    pred_nodes = [{"id": "p1", "bbox": [500, 500, 50, 50]}]

    m = match_nodes_center_distance(gt_nodes, pred_nodes, alpha=0.2)
    assert m == {}


def test_one_gt_only_matches_one_pred():
    gt_nodes = [{"id": "g1", "bbox": [0, 0, 100, 100]}]
    pred_nodes = [
        {"id": "p1", "bbox": [10, 10, 80, 80]},
        {"id": "p2", "bbox": [12, 12, 80, 80]},
    ]

    m = match_nodes_center_distance(gt_nodes, pred_nodes, alpha=0.5)
    assert len(m) == 1
    assert list(m.values()) == ["g1"]


def test_project_edges_drops_unmatched():
    pred_edges = [
        {"from": "p1", "to": "p2"},
        {"from": "p1", "to": "px"},
    ]
    pred_to_gt = {"p1": "g1", "p2": "g2"}

    projected = project_pred_edges_to_gt(pred_edges, pred_to_gt)
    assert projected == {("g1", "g2")}


def test_invalid_alpha():
    with pytest.raises(MatchingError):
        match_nodes_center_distance([], [], alpha=0)
