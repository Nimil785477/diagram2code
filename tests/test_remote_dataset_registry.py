from __future__ import annotations

import pytest

from diagram2code.datasets.fetching.descriptors import DatasetDescriptor
from diagram2code.datasets.fetching.errors import DatasetNotFoundError
from diagram2code.datasets.fetching.registry import RemoteDatasetRegistry


def test_registry_list_sorted() -> None:
    r = RemoteDatasetRegistry(
        _items={
            "b": DatasetDescriptor(name="b", version="1", description="x"),
            "a": DatasetDescriptor(name="a", version="1", description="x"),
        }
    )
    assert r.list() == ["a", "b"]


def test_registry_get_missing_raises() -> None:
    r = RemoteDatasetRegistry.builtins()
    with pytest.raises(DatasetNotFoundError):
        _ = r.get("does_not_exist")


def test_registry_builtins_includes_flowlearn() -> None:
    r = RemoteDatasetRegistry.builtins()
    assert "flowlearn" in r.list()
