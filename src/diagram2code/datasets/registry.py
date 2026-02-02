from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .loader import load_dataset
from .types import Dataset
from .validation import DatasetError


def _default_config_path() -> Path:
    # Cross-platform, deterministic:
    # Windows: C:\Users\<user>\.diagram2code\datasets.json
    # Unix:    ~/.diagram2code/datasets.json
    return Path.home() / ".diagram2code" / "datasets.json"


def _load_mapping_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"Invalid JSON in dataset registry file: {path}") from exc

    if not isinstance(raw, dict):
        raise DatasetError(f"Dataset registry file must be a JSON object: {path}")

    out: dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k] = v
    return out


def _load_env_mapping() -> dict[str, str]:
    val = os.environ.get("DIAGRAM2CODE_DATASET_PATHS")
    if not val:
        return {}

    # Accept either a JSON object string or a path to a JSON file
    maybe_path = Path(val)
    if maybe_path.exists():
        return _load_mapping_file(maybe_path)

    try:
        raw = json.loads(val)
    except json.JSONDecodeError as exc:
        raise DatasetError(
            "DIAGRAM2CODE_DATASET_PATHS must be a JSON object string or a path to a JSON file"
        ) from exc

    if not isinstance(raw, dict):
        raise DatasetError("DIAGRAM2CODE_DATASET_PATHS must be a JSON object")

    out: dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k] = v
    return out


def _example_dataset_root(example_name: str) -> Path:
    """
    Return the root directory of a built-in example dataset.
    Uses importlib.resources so it works from source + installed.
    """
    from importlib import resources

    # package: diagram2code.datasets.examples
    base = resources.files("diagram2code.datasets.examples")
    root = base / example_name
    # Convert to filesystem path (works for editable installs and normal installs)
    return Path(root)


@dataclass(frozen=True)
class DatasetRegistry:
    """
    Resolves dataset references to dataset root directories.

    Resolution order:
    1) If ref exists as a path -> that path
    2) Built-in examples: "example:<name>" (e.g. example:minimal_v1)
    3) Env var mapping: DIAGRAM2CODE_DATASET_PATHS (JSON object or JSON file path)
    4) User file mapping: ~/.diagram2code/datasets.json
    """

    config_path: Path = _default_config_path()

    def resolve_root(self, ref: str | Path) -> Path:
        ref_path = Path(ref)
        if ref_path.exists():
            return ref_path

        ref_str = str(ref)

        if ref_str.startswith("example:"):
            example_name = ref_str.split(":", 1)[1].strip()
            if not example_name:
                raise DatasetError("Example dataset reference must be 'example:<name>'")
            root = _example_dataset_root(example_name)
            if not root.exists():
                raise DatasetError(f"Unknown example dataset: {example_name}")
            return root

        env_map = _load_env_mapping()
        if ref_str in env_map:
            root = Path(env_map[ref_str])
            if not root.exists():
                raise DatasetError(
                    f"Dataset path from env mapping does not exist for {ref_str!r}: {root}"
                )
            return root

        file_map = _load_mapping_file(self.config_path)
        if ref_str in file_map:
            root = Path(file_map[ref_str])
            if not root.exists():
                raise DatasetError(
                    f"Dataset path from registry file does not exist for {ref_str!r}: {root}"
                )
            return root

        raise DatasetError(
            f"Unknown dataset reference: {ref_str!r}. "
            "Provide a path, use example:<name>, set DIAGRAM2CODE_DATASET_PATHS, "
            "or add it to ~/.diagram2code/datasets.json"
        )

    def load(self, ref: str | Path, *, validate_graphs: bool = True) -> Dataset:
        root = self.resolve_root(ref)
        return load_dataset(root, validate_graphs=validate_graphs)


def resolve_dataset(ref: str | Path, *, validate_graphs: bool = True) -> Dataset:
    """
    Convenience API: resolve via default registry and load.
    """
    return DatasetRegistry().load(ref, validate_graphs=validate_graphs)
