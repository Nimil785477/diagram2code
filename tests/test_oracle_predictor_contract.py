import json
from pathlib import Path

from diagram2code.datasets import load_dataset
from diagram2code.predictors.oracle import OraclePredictor


def _write_min_phase3_dataset(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "graphs").mkdir(parents=True, exist_ok=True)

    # dataset.json (Phase-3 requires schema_version)
    dataset_json = {
        "schema_version": "1.0",
        "name": "test-ds",
        "version": "1.0",
        "splits": {"test": ["s1"]},
    }
    (root / "dataset.json").write_text(json.dumps(dataset_json), encoding="utf-8")

    # image placeholder (loader checks existence; content not required)
    (root / "images" / "s1.png").write_bytes(b"")

    # graph
    graph = {
        "nodes": [{"id": "n1", "bbox": [0, 0, 10, 10]}],
        "edges": [{"source": "n1", "target": "n1"}],
    }
    (root / "graphs" / "s1.json").write_text(json.dumps(graph), encoding="utf-8")


def test_oracle_predictor_returns_nodes_edges(tmp_path: Path):
    ds_root = tmp_path / "phase3_ds"
    _write_min_phase3_dataset(ds_root)

    ds = load_dataset(ds_root)
    sample = ds.samples("test")[0]

    pred = OraclePredictor().predict(sample)

    assert "nodes" in pred
    assert "edges" in pred
    assert isinstance(pred["nodes"], list)
    assert isinstance(pred["edges"], list)

    if pred["nodes"]:
        n0 = pred["nodes"][0]
        assert "id" in n0
        assert "bbox" in n0
