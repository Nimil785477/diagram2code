from __future__ import annotations

from diagram2code.datasets import load_dataset
from diagram2code.datasets.synthflow import build_synthflow_dataset
from diagram2code.predictors.naive import NaivePredictor


def test_naive_predictor_returns_single_centered_node(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    build_synthflow_dataset(out=dataset_dir, split="test", num_samples=1, seed=0)

    ds = load_dataset(dataset_dir)
    sample = next(iter(ds.samples("test")))

    pred = NaivePredictor().predict(sample)

    assert set(pred.keys()) == {"nodes", "edges"}
    assert len(pred["nodes"]) == 1
    assert pred["edges"] == []

    node = pred["nodes"][0]
    assert node["id"] == "0"
    assert isinstance(node["bbox"], list)
    assert len(node["bbox"]) == 4
