from __future__ import annotations

import json
from pathlib import Path

import pytest

from diagram2code.benchmark.predictor import PredGraph
from diagram2code.benchmark.runner import run_benchmark


def _write_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    images_dir = root / "images"
    graphs_dir = root / "graphs"
    images_dir.mkdir(parents=True)
    graphs_dir.mkdir(parents=True)

    # Minimal valid dataset.json
    (root / "dataset.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "name": "test-ds",
                "version": "0.1",
                "splits": {"all": ["0001"]},
            }
        ),
        encoding="utf-8",
    )

    # Write a tiny image (we don't actually read pixels in runner tests)
    (images_dir / "0001.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    # Phase 3 graph schema: node ids are strings, edges use source/target
    (graphs_dir / "0001.json").write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "1", "bbox": [50, 50, 50, 50]},
                    {"id": "2", "bbox": [200, 50, 50, 50]},
                ],
                "edges": [{"source": "1", "target": "2"}],
            }
        ),
        encoding="utf-8",
    )

    return root


def test_run_benchmark_perfect_prediction(tmp_path: Path) -> None:
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
    assert result.aggregate.node.f1 == 1.0
    assert result.aggregate.edge.f1 == 1.0
    assert result.aggregate.direction_accuracy == 1.0
    assert result.aggregate.exact_match_rate == 1.0


def test_run_benchmark_wrong_edge_direction(tmp_path: Path) -> None:
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

    assert len(result.samples) == 1
    assert result.aggregate.node.f1 == 1.0
    assert result.aggregate.edge.f1 == 0.0
    assert result.aggregate.direction_accuracy == 0.0
    assert result.aggregate.exact_match_rate == 0.0


def test_run_benchmark_empty_dataset_raises(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()

    from diagram2code.datasets import DatasetError

    with pytest.raises(DatasetError):
        run_benchmark(
            dataset_path=root,
            predictor=lambda _: PredGraph(nodes=[], edges=[]),
            alpha=0.2,
        )
