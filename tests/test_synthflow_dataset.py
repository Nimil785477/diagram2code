from __future__ import annotations

import json

from diagram2code.datasets.synthflow import build_synthflow_dataset


def test_build_synthflow_dataset_v2_writes_expected_files(tmp_path):
    out = tmp_path / "synthflow"

    build_synthflow_dataset(out=out, split="test", num_samples=5, seed=0)

    assert (out / "dataset.json").exists()
    assert (out / "splits.json").exists()
    assert (out / "images").exists()
    assert (out / "graphs").exists()

    dataset = json.loads((out / "dataset.json").read_text(encoding="utf-8"))
    splits = json.loads((out / "splits.json").read_text(encoding="utf-8"))

    assert dataset["name"] == "synthflow_v2"
    assert dataset["version"] == "2"
    assert dataset["extra"]["generator"] == "synthflow_v2"
    assert dataset["extra"]["num_samples"] == 5
    assert dataset["extra"]["seed"] == 0

    sample_ids = splits["splits"]["test"]
    assert len(sample_ids) == 5

    for sample_id in sample_ids:
        image_path = out / "images" / f"{sample_id}.png"
        graph_path = out / "graphs" / f"{sample_id}.json"

        assert image_path.exists()
        assert graph_path.exists()

        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        assert "nodes" in graph
        assert "edges" in graph
        assert len(graph["nodes"]) >= 3
        assert len(graph["edges"]) >= 2

        for node in graph["nodes"]:
            assert set(node.keys()) == {"id", "bbox"}
            assert isinstance(node["bbox"], list)
            assert len(node["bbox"]) == 4

        for edge in graph["edges"]:
            assert set(edge.keys()) == {"source", "target"}
