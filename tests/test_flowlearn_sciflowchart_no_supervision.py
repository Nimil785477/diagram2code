from __future__ import annotations

import json
from pathlib import Path

import pytest

from diagram2code.datasets.adapters.flowlearn import convert_flowlearn
from diagram2code.datasets.validation import DatasetError


def test_flowlearn_sciflowchart_no_supervision(tmp_path: Path) -> None:
    """
    The installed HF FlowLearn snapshot (as observed) contains SciFlowchart images and
    figure metadata JSON, but no graph supervision (Mermaid / nodes / edges).

    Conversion to Phase-3 (images/ + graphs/) must fail fast with a helpful DatasetError.
    """
    flowlearn_root = tmp_path / "FlowLearn"

    # Minimal SciFlowchart structure
    sci = flowlearn_root / "SciFlowchart"
    images = sci / "images"
    images.mkdir(parents=True)

    # One dummy image file
    (images / "paper-Figure1-1.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    # Split JSON: figure metadata only (no mermaid/nodes/edges)
    records = [
        {
            "image_file": "paper-Figure1-1.png",
            "caption": "Fig. 1: A flowchart of something (caption text only).",
            "regionBoundary": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            "imageText": ["Boxes"],
        }
    ]
    (sci / "test.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

    out = tmp_path / "out"

    with pytest.raises(DatasetError) as ei:
        convert_flowlearn(
            flowlearn_root=flowlearn_root,
            subset="SciFlowchart",
            split="test",
            out=out,
            limit=1,
            strict=True,
        )

    msg = str(ei.value).lower()
    assert "sciflowchart" in msg
    assert "no graph" in msg or "graph annotations" in msg or "cannot generate" in msg
