from __future__ import annotations

from pathlib import Path

import pytest

from diagram2code.datasets.fetching.cache import dataset_dir, get_cache_root


def test_get_cache_root_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DIAGRAM2CODE_CACHE_DIR", str(tmp_path))
    root = get_cache_root()
    assert root == tmp_path.resolve() / "datasets"


def test_dataset_dir_joins_name_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DIAGRAM2CODE_CACHE_DIR", str(tmp_path))
    p = dataset_dir("flowlearn", "1.0.0")
    assert p == tmp_path.resolve() / "datasets" / "flowlearn" / "1.0.0"


@pytest.mark.parametrize("name,version", [("", "1"), ("x", ""), ("   ", "1"), ("x", "   ")])
def test_dataset_dir_rejects_empty(name: str, version: str) -> None:
    with pytest.raises(ValueError):
        _ = dataset_dir(name, version)
