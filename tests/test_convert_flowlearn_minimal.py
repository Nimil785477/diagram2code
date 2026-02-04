from __future__ import annotations

import json
from pathlib import Path

from diagram2code.datasets import load_dataset
from diagram2code.datasets.adapters.flowlearn import convert_flowlearn


def test_convert_flowlearn_minimal(tmp_path: Path) -> None:
    flow = tmp_path / "FlowLearn"
    subset = flow / "SimFlowchart"
    (subset / "images").mkdir(parents=True)

    (subset / "images" / "img1.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    rec = {
        "id": "sample-1",
        "image": "img1.png",
        "nodes": [
            {"id": "A", "bbox": [10, 10, 50, 40]},
            {"id": "B", "bbox": [100, 10, 50, 40]},
        ],
        "edges": [{"source": "A", "target": "B"}],
    }
    (subset / "test.json").write_text(json.dumps([rec]), encoding="utf-8")

    out_root = tmp_path / "out_ds"
    convert_flowlearn(
        flowlearn_root=flow,
        subset="SimFlowchart",
        split="test",
        out_root=out_root,
    )

    ds = load_dataset(out_root)
    assert ds.metadata.name == "flowlearn-simflowchart"

    samples = list(ds.samples("test"))
    assert len(samples) == 1
    g = samples[0].load_graph_json()
    assert len(g["nodes"]) == 2
    assert len(g["edges"]) == 1
