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
        _ = r.get("flowlearn")
