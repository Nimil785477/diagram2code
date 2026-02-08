from __future__ import annotations

from pathlib import Path


def get_cache_root() -> Path:
    """Implemented in Step 2."""
    raise NotImplementedError


def dataset_dir(name: str, version: str) -> Path:
    """Implemented in Step 2."""
    raise NotImplementedError
