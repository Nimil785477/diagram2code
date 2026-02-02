from __future__ import annotations

import json
from pathlib import Path

import pytest

from diagram2code.datasets import DatasetError, load_dataset


def _write_png(path: Path) -> None:
    # Small deterministic image; avoids depending on project CV code.
    from PIL import Image

    img = Image.new("RGB", (10, 10))
    img.save(path)


def test_load_dataset_happy_path_no_splits(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    (root / "images").mkdir(parents=True)
    (root / "graphs").mkdir(parents=True)

    meta = {"schema_version": "1.0", "name": "ds", "version": "0.1"}
    (root / "dataset.json").write_text(json.dumps(meta), encoding="utf-8")

    _write_png(root / "images" / "0001.png")
    (root / "graphs" / "0001.json").write_text(
        json.dumps({"nodes": [], "edges": []}),
        encoding="utf-8",
    )

    ds = load_dataset(root)
    samples = list(ds.samples())

    assert ds.metadata.name == "ds"
    assert len(samples) == 1
    assert samples[0].sample_id == "0001"
    assert samples[0].split == "all"
    assert samples[0].load_graph_json() == {"nodes": [], "edges": []}


def test_load_dataset_happy_path_with_splits(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    (root / "images").mkdir(parents=True)
    (root / "graphs").mkdir(parents=True)

    meta = {
        "schema_version": "1.0",
        "name": "ds",
        "version": "0.1",
        "splits": {"train": ["0001"], "test": ["0002"]},
    }
    (root / "dataset.json").write_text(json.dumps(meta), encoding="utf-8")

    _write_png(root / "images" / "0001.png")
    (root / "graphs" / "0001.json").write_text(
        json.dumps({"nodes": [], "edges": []}),
        encoding="utf-8",
    )

    _write_png(root / "images" / "0002.png")
    (root / "graphs" / "0002.json").write_text(
        json.dumps({"nodes": [], "edges": []}),
        encoding="utf-8",
    )
    ds = load_dataset(root)
    assert set(ds.splits()) == {"test", "train"}
    assert [s.sample_id for s in ds.samples("train")] == ["0001"]
    assert [s.sample_id for s in ds.samples("test")] == ["0002"]


def test_load_dataset_mismatch_ids_raises(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    (root / "images").mkdir(parents=True)
    (root / "graphs").mkdir(parents=True)
    (root / "dataset.json").write_text(json.dumps({"schema_version": "1.0"}), encoding="utf-8")

    _write_png(root / "images" / "0001.png")
    # missing graph

    with pytest.raises(DatasetError) as e:
        load_dataset(root)

    assert "mismatch" in str(e.value).lower()


def test_load_dataset_split_unknown_id_raises(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    (root / "images").mkdir(parents=True)
    (root / "graphs").mkdir(parents=True)

    meta = {"schema_version": "1.0", "splits": {"train": ["nope"]}}
    (root / "dataset.json").write_text(json.dumps(meta), encoding="utf-8")

    _write_png(root / "images" / "0001.png")
    (root / "graphs" / "0001.json").write_text(
        json.dumps({"nodes": [], "edges": []}),
        encoding="utf-8",
    )

    with pytest.raises(DatasetError) as e:
        load_dataset(root)

    assert "unknown" in str(e.value).lower()
