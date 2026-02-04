from __future__ import annotations

from pathlib import Path

from diagram2code.datasets import load_dataset
from diagram2code.datasets.synthetic_basic import generate_synthetic_basic


def test_generate_synthetic_basic_dataset_loads(tmp_path: Path) -> None:
    root = tmp_path / "synthetic_basic"
    generate_synthetic_basic(root, n=3, seed=0, split="test")

    ds = load_dataset(root)
    samples = list(ds.samples("test"))

    assert ds.metadata.schema_version == "1.0"
    assert ds.metadata.name == "synthetic-basic"
    assert len(samples) == 3

    s = samples[0]
    assert s.image_path.exists()
    assert s.graph_path.exists()

    graph = s.load_graph_json()
    assert "nodes" in graph
    assert "edges" in graph
