from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_cache_dir

_ENV_CACHE_DIR = "DIAGRAM2CODE_CACHE_DIR"


def get_cache_root() -> Path:
    """
    Return the datasets cache root directory for diagram2code.

    Override with env var:
      DIAGRAM2CODE_CACHE_DIR=/path/to/cache

    Contract:
      get_cache_root() returns the directory that contains dataset names.

    Layout:
      {cache_root}/{name}/{version}

    Default:
      platformdirs.user_cache_dir("diagram2code") / "datasets"
    """
    override = os.environ.get(_ENV_CACHE_DIR)
    if override:
        return Path(override).expanduser().resolve() / "datasets"

    return Path(user_cache_dir("diagram2code")) / "datasets"


def dataset_dir(name: str, version: str) -> Path:
    """
    Canonical install directory for a specific dataset version.

    Layout:
      {cache_root}/{name}/{version}
    """
    name = name.strip()
    version = version.strip()
    if not name:
        raise ValueError("dataset name must be non-empty")
    if not version:
        raise ValueError("dataset version must be non-empty")

    return get_cache_root() / name / version
