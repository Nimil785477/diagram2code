from __future__ import annotations

import json
from pathlib import Path

import pytest

from diagram2code.datasets import DatasetError, load_dataset


def _write_png(path: Path) -> None:
    from PIL import Image

    img = Image.new("RGB", (10, 10))
    img.save(path)


def _write_graph(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _mk_base_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    (root / "images").mkdir(parents=True)
    (root / "graphs").mkdir(parents=True)
    return root


def test_dataset_missing_schema_version_raises(tmp_path: Path) -> None:
    root = _mk_base_dataset(tmp_path)
    (root / "dataset.json").write_text(json.dumps({"name": "ds"}), encoding="utf-8")

    _write_png(root / "images" / "0001.png")
    _write_graph(
        root / "graphs" / "0001.json",
        {"nodes": [{"id": "n1"}], "edges": []},
    )

    with pytest.raises(DatasetError) as exc:
        load_dataset(root)

    assert "schema_version" in str(exc.value)


def test_dataset_unsupported_schema_version_raises(tmp_path: Path) -> None:
    root = _mk_base_dataset(tmp_path)
    meta = {"schema_version": "9.9", "name": "ds", "version": "0.1"}
    (root / "dataset.json").write_text(json.dumps(meta), encoding="utf-8")

    _write_png(root / "images" / "0001.png")
    _write_graph(
        root / "graphs" / "0001.json",
        {"nodes": [{"id": "n1"}], "edges": []},
    )

    with pytest.raises(DatasetError) as exc:
        load_dataset(root)

    assert "unsupported" in str(exc.value).lower()


def test_graph_invalid_json_raises(tmp_path: Path) -> None:
    root = _mk_base_dataset(tmp_path)
    meta = {"schema_version": "1.0", "name": "ds", "version": "0.1"}
    (root / "dataset.json").write_text(json.dumps(meta), encoding="utf-8")

    _write_png(root / "images" / "0001.png")
    (root / "graphs" / "0001.json").write_text("{not-json", encoding="utf-8")

    with pytest.raises(DatasetError) as exc:
        load_dataset(root)

    assert "invalid json" in str(exc.value).lower()


def test_graph_missing_nodes_edges_raises(tmp_path: Path) -> None:
    root = _mk_base_dataset(tmp_path)
    meta = {"schema_version": "1.0", "name": "ds", "version": "0.1"}
    (root / "dataset.json").write_text(json.dumps(meta), encoding="utf-8")

    _write_png(root / "images" / "0001.png")
    _write_graph(root / "graphs" / "0001.json", {"foo": 1})

    with pytest.raises(DatasetError) as exc:
        load_dataset(root)

    msg = str(exc.value).lower()
    assert "nodes" in msg
    assert "edges" in msg


def test_graph_node_missing_id_raises(tmp_path: Path) -> None:
    root = _mk_base_dataset(tmp_path)
    meta = {"schema_version": "1.0", "name": "ds", "version": "0.1"}
    (root / "dataset.json").write_text(json.dumps(meta), encoding="utf-8")

    _write_png(root / "images" / "0001.png")
    _write_graph(
        root / "graphs" / "0001.json",
        {"nodes": [{}], "edges": []},
    )

    with pytest.raises(DatasetError) as exc:
        load_dataset(root)

    assert "missing valid 'id'" in str(exc.value).lower()


def test_graph_edge_missing_source_target_raises(tmp_path: Path) -> None:
    root = _mk_base_dataset(tmp_path)
    meta = {"schema_version": "1.0", "name": "ds", "version": "0.1"}
    (root / "dataset.json").write_text(json.dumps(meta), encoding="utf-8")

    _write_png(root / "images" / "0001.png")
    _write_graph(
        root / "graphs" / "0001.json",
        {
            "nodes": [{"id": "n1"}, {"id": "n2"}],
            "edges": [{"source": "n1"}],
        },
    )

    with pytest.raises(DatasetError) as exc:
        load_dataset(root)

    assert "missing valid 'target'" in str(exc.value).lower()
