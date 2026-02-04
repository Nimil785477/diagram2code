from __future__ import annotations

import json
from pathlib import Path

from diagram2code.datasets import load_dataset
from diagram2code.datasets.adapters.flowlearn import convert_flowlearn


def test_convert_flowlearn_simflowchart_char_minimal(tmp_path: Path) -> None:
    flow = tmp_path / "FlowLearn"
    images_dir = flow / "SimFlowchart" / "images" / "char_TextOCR" / "images"
    split_dir = flow / "SimFlowchart" / "char"
    images_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    # fake image
    (images_dir / "0.jpeg").write_bytes(b"\xff\xd8\xff\xd9")  # minimal JPEG header/footer

    rec = {
        "file": "0.jpeg",
        "mermaid": "```mermaid\nflowchart LR\nentity0(A)\nentity1(B)\nentity0 --> entity1\n```",
        "caption": "A points to B.",
        "ocr": "A, B",
        "meta": {
            "text": {
                "0": {
                    "x0": 10.0,
                    "y0": 20.0,
                    "x1": 30.0,
                    "y1": 50.0,
                    "text": "A",
                    "mermaid_entity_i": 0,
                },
                "1": {
                    "x0": 60.0,
                    "y0": 20.0,
                    "x1": 90.0,
                    "y1": 50.0,
                    "text": "B",
                    "mermaid_entity_i": 1,
                },
            }
        },
    }
    (split_dir / "test.json").write_text(json.dumps([rec]), encoding="utf-8")

    out = tmp_path / "out_ds"
    convert_flowlearn(
        flowlearn_root=flow,
        subset="SimFlowchart/char",
        split="test",
        out=out,
        limit=1,
        strict=True,
    )

    ds = load_dataset(out)
    assert ds.metadata.name == "flowlearn_simflowchart_char"
    assert "test" in ds.metadata.splits
    assert len(ds.items) == 1
