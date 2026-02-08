from __future__ import annotations

from .descriptors import DatasetDescriptor


class RemoteDatasetRegistry:
    """Implemented in Step 3/6 (starts as in-memory dict)."""

    def list(self) -> list[str]:
        raise NotImplementedError

    def get(self, name: str) -> DatasetDescriptor:
        raise NotImplementedError
