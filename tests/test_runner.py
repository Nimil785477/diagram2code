from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from diagram2code.benchmark.predictor import PredGraph
from diagram2code.benchmark.runner import run_benchmark


def _write_dataset(tmp_path: Path) -> Path:
    ds = tmp_path / "ds"
    (ds / "images").mkdir(parents=True)
    (ds / "graphs").mkdir(parents=True)

    img_path = ds / "images" / "sample.png"
    Image.new("RGB", (400, 300), "white").save(img_path)

    graph = {
        "nodes": [
            {"id": 1, "bbox": [50, 50, 50, 50]},
            {"id": 2, "bbox": [200, 50, 50, 50]},
        ],
        "edges": [{"from": 1, "to": 2}],
    }

    (ds / "graphs" / "sample.graph.json").write_text(json.dumps(graph), encoding="utf-8")
    return ds


def test_run_benchmark_perfect_prediction(tmp_path: Path):
    ds = _write_dataset(tmp_path)

    def predictor(_image_path: Path) -> PredGraph:
        return PredGraph(
            nodes=[
                {"id": "p1", "bbox": [50, 50, 50, 50]},
                {"id": "p2", "bbox": [200, 50, 50, 50]},
            ],
            edges=[{"from": "p1", "to": "p2"}],
        )

    result = run_benchmark(dataset_path=ds, predictor=predictor, alpha=0.2)

    assert len(result.samples) == 1
    s = result.samples[0].metrics
    assert s.node.f1 == 1.0
    assert s.edge.f1 == 1.0
    assert s.direction_accuracy == 1.0
    assert s.exact_match

    agg = result.aggregate
    assert agg.node.f1 == 1.0
    assert agg.edge.f1 == 1.0
    assert agg.exact_match_rate == 1.0


def test_run_benchmark_wrong_edge_direction(tmp_path: Path):
    ds = _write_dataset(tmp_path)

    def predictor(_image_path: Path) -> PredGraph:
        return PredGraph(
            nodes=[
                {"id": "p1", "bbox": [50, 50, 50, 50]},
                {"id": "p2", "bbox": [200, 50, 50, 50]},
            ],
            edges=[{"from": "p2", "to": "p1"}],  # reversed
        )

    result = run_benchmark(dataset_path=ds, predictor=predictor, alpha=0.2)
    s = result.samples[0].metrics

    assert s.node.f1 == 1.0
    assert s.edge.f1 == 0.0
    assert s.direction_accuracy == 0.0
    assert not s.exact_match
