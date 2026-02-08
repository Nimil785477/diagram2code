from __future__ import annotations

from pathlib import Path

from .descriptors import DatasetDescriptor


def fetch_dataset(
    descriptor: DatasetDescriptor,
    cache_root: Path | None = None,
    force: bool = False,
) -> Path:
    """Implemented in Step 3."""
    raise NotImplementedError
