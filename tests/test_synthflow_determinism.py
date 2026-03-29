from __future__ import annotations

import hashlib

from diagram2code.datasets.synthflow import build_synthflow_dataset


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_synthflow_dataset_v2_is_deterministic(tmp_path):
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"

    build_synthflow_dataset(out=out1, split="test", num_samples=8, seed=123)
    build_synthflow_dataset(out=out2, split="test", num_samples=8, seed=123)

    assert (out1 / "dataset.json").read_text(encoding="utf-8") == (out2 / "dataset.json").read_text(
        encoding="utf-8"
    )
    assert (out1 / "splits.json").read_text(encoding="utf-8") == (out2 / "splits.json").read_text(
        encoding="utf-8"
    )

    graphs1 = sorted((out1 / "graphs").glob("*.json"))
    graphs2 = sorted((out2 / "graphs").glob("*.json"))
    images1 = sorted((out1 / "images").glob("*.png"))
    images2 = sorted((out2 / "images").glob("*.png"))

    assert [p.name for p in graphs1] == [p.name for p in graphs2]
    assert [p.name for p in images1] == [p.name for p in images2]

    for p1, p2 in zip(graphs1, graphs2, strict=True):
        assert p1.read_text(encoding="utf-8") == p2.read_text(encoding="utf-8")

    for p1, p2 in zip(images1, images2, strict=True):
        assert _sha256(p1) == _sha256(p2)
