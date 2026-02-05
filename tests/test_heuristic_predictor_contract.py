import json
from pathlib import Path

from diagram2code.datasets import load_dataset
from diagram2code.predictors.heuristic import HeuristicPredictor


def _write_min_phase3_dataset(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "graphs").mkdir(parents=True, exist_ok=True)

    dataset_json = {
        "schema_version": "1.0",
        "name": "test-ds",
        "version": "1.0",
        "splits": {"test": ["s1"]},
    }
    (root / "dataset.json").write_text(json.dumps(dataset_json), encoding="utf-8")

    # image placeholder (exists only)
    (root / "images" / "s1.png").write_bytes(b"")

    # provide a few nodes so heuristic can create edges
    graph = {
        "nodes": [
            {"id": "n1", "bbox": [0, 0, 10, 10]},
            {"id": "n2", "bbox": [0, 20, 10, 10]},
            {"id": "n3", "bbox": [0, 40, 10, 10]},
        ],
        "edges": [],
    }
    (root / "graphs" / "s1.json").write_text(json.dumps(graph), encoding="utf-8")


def test_heuristic_predictor_returns_valid_graph(tmp_path: Path):
    ds_root = tmp_path / "phase3_ds"
    _write_min_phase3_dataset(ds_root)

    ds = load_dataset(ds_root)
    sample = ds.samples("test")[0]

    pred = HeuristicPredictor().predict(sample)

    assert "nodes" in pred
    assert "edges" in pred

    assert len(pred["nodes"]) == 3
    # heuristic connects sequentially => 2 edges
    assert len(pred["edges"]) == 2

    for n in pred["nodes"]:
        assert "id" in n
        assert "bbox" in n

    for e in pred["edges"]:
        assert "source" in e
        assert "target" in e
