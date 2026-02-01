from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from diagram2code.benchmark.dataset import DatasetValidationError, load_dataset


def _write_img(path: Path, w: int = 200, h: int = 120) -> None:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _write_gt(path: Path, *, nodes, edges) -> None:
    path.write_text(json.dumps({"nodes": nodes, "edges": edges}, indent=2), encoding="utf-8")


def test_load_dataset_happy_path(tmp_path: Path):
    ds = tmp_path / "synthetic_basic"
    (ds / "images").mkdir(parents=True)
    (ds / "graphs").mkdir(parents=True)

    img = ds / "images" / "img_0001.png"
    _write_img(img, 200, 120)

    gt = ds / "graphs" / "img_0001.graph.json"
    _write_gt(
        gt,
        nodes=[{"id": 0, "bbox": [10, 10, 50, 30]}, {"id": 1, "bbox": [120, 40, 50, 30]}],
        edges=[{"from": 0, "to": 1}],
    )

    dataset = load_dataset(ds)
    assert dataset.name == "synthetic_basic"
    assert len(dataset.items) == 1
    assert dataset.items[0].image_path.name == "img_0001.png"
    assert dataset.items[0].graph_path.name == "img_0001.graph.json"
    assert len(dataset.items[0].gt.nodes) == 2
    assert len(dataset.items[0].gt.edges) == 1


def test_missing_graph_raises(tmp_path: Path):
    ds = tmp_path / "ds"
    (ds / "images").mkdir(parents=True)
    (ds / "graphs").mkdir(parents=True)

    img = ds / "images" / "img_0001.png"
    _write_img(img)

    with pytest.raises(DatasetValidationError, match="Missing graph"):
        load_dataset(ds)


def test_invalid_bbox_out_of_bounds_raises(tmp_path: Path):
    ds = tmp_path / "ds"
    (ds / "images").mkdir(parents=True)
    (ds / "graphs").mkdir(parents=True)

    img = ds / "images" / "img_0001.png"
    _write_img(img, 100, 100)

    gt = ds / "graphs" / "img_0001.graph.json"
    _write_gt(
        gt,
        nodes=[{"id": 0, "bbox": [80, 80, 50, 30]}],  # out of bounds
        edges=[],
    )

    with pytest.raises(DatasetValidationError, match="bbox out of bounds"):
        load_dataset(ds)


def test_edge_references_unknown_node_raises(tmp_path: Path):
    ds = tmp_path / "ds"
    (ds / "images").mkdir(parents=True)
    (ds / "graphs").mkdir(parents=True)

    img = ds / "images" / "img_0001.png"
    _write_img(img)

    gt = ds / "graphs" / "img_0001.graph.json"
    _write_gt(
        gt,
        nodes=[{"id": 0, "bbox": [10, 10, 20, 20]}],
        edges=[{"from": 0, "to": 1}],  # invalid
    )

    with pytest.raises(DatasetValidationError, match="unknown node"):
        load_dataset(ds)
