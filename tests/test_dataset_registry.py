from __future__ import annotations

import json
from pathlib import Path

import pytest

from diagram2code.datasets import DatasetError, DatasetRegistry, resolve_dataset


def test_registry_resolves_example_dataset() -> None:
    ds = resolve_dataset("example:minimal_v1")
    samples = list(ds.samples())
    assert len(samples) == 1
    assert samples[0].sample_id == "0001"
    assert samples[0].graph_path.name == "0001.json"
    assert samples[0].image_path.suffix.lower() == ".svg"


def test_registry_resolves_path(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    (root / "images").mkdir(parents=True)
    (root / "graphs").mkdir(parents=True)

    (root / "dataset.json").write_text(
        json.dumps({"schema_version": "1.0", "name": "ds", "version": "0.1"}),
        encoding="utf-8",
    )
    (root / "images" / "0001.svg").write_text("<svg/>", encoding="utf-8")
    (root / "graphs" / "0001.json").write_text(
        json.dumps({"nodes": [{"id": "A"}], "edges": []}),
        encoding="utf-8",
    )

    reg = DatasetRegistry()
    ds = reg.load(root)
    assert [s.sample_id for s in ds.samples()] == ["0001"]


def test_registry_env_mapping(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "ds"
    (root / "images").mkdir(parents=True)
    (root / "graphs").mkdir(parents=True)

    (root / "dataset.json").write_text(
        json.dumps({"schema_version": "1.0", "name": "ds", "version": "0.1"}),
        encoding="utf-8",
    )
    (root / "images" / "0001.svg").write_text("<svg/>", encoding="utf-8")
    (root / "graphs" / "0001.json").write_text(
        json.dumps({"nodes": [{"id": "A"}], "edges": []}),
        encoding="utf-8",
    )

    monkeypatch.setenv(
        "DIAGRAM2CODE_DATASET_PATHS",
        json.dumps({"myds": str(root)}),
    )

    reg = DatasetRegistry()
    ds = reg.load("myds")
    assert [s.sample_id for s in ds.samples()] == ["0001"]


def test_registry_unknown_ref_raises() -> None:
    reg = DatasetRegistry()
    with pytest.raises(DatasetError):
        reg.resolve_root("does-not-exist-xyz")
