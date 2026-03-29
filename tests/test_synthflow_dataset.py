from __future__ import annotations

import json
from pathlib import Path

from diagram2code.datasets.loader import load_dataset
from diagram2code.datasets.synthflow import build_synthflow_dataset


def test_build_synthflow_dataset(tmp_path: Path) -> None:
    out = tmp_path / "synthflow_ds"

    build_synthflow_dataset(
        out=out,
        split="test",
        num_samples=4,
        seed=0,
    )

    assert (out / "dataset.json").exists()
    assert (out / "splits.json").exists()
    assert (out / "images").is_dir()
    assert (out / "graphs").is_dir()

    graph_files = sorted((out / "graphs").glob("*.json"))
    image_files = sorted((out / "images").glob("*.png"))

    assert len(graph_files) == 4
    assert len(image_files) == 4

    graph = json.loads(graph_files[0].read_text(encoding="utf-8"))
    assert "nodes" in graph
    assert "edges" in graph
    assert len(graph["nodes"]) >= 2
    assert len(graph["edges"]) >= 1

    for node in graph["nodes"]:
        assert isinstance(node["id"], str)
        assert isinstance(node["bbox"], list)
        assert len(node["bbox"]) == 4
        assert all(isinstance(v, int) for v in node["bbox"])

    for edge in graph["edges"]:
        assert isinstance(edge["source"], str)
        assert isinstance(edge["target"], str)

    ds = load_dataset(out)
    assert len(list(ds.samples())) == 4
