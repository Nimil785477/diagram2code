from __future__ import annotations

from dataclasses import dataclass

from .builtins import built_in_descriptors
from .descriptors import DatasetDescriptor
from .errors import DatasetNotFoundError


@dataclass(frozen=True, slots=True)
class RemoteDatasetRegistry:
    _items: dict[str, DatasetDescriptor]

    @classmethod
    def builtins(cls) -> RemoteDatasetRegistry:
        return cls(_items=built_in_descriptors())

    def list(self) -> list[str]:
        return sorted(self._items.keys())

    def get(self, name: str) -> DatasetDescriptor:
        key = name.strip()
        if not key:
            raise DatasetNotFoundError("Dataset name must be non-empty")
        try:
            return self._items[key]
        except KeyError as e:
            raise DatasetNotFoundError(f"Unknown remote dataset: {key}") from e
