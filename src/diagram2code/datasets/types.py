from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetMetadata:
    schema_version: str
    name: str
    version: str
    splits: Mapping[str, tuple[str, ...]]  # e.g. {"train": ("0001", ...)}
    extra: Mapping[str, Any]


@dataclass(frozen=True)
class DatasetSample:
    sample_id: str
    image_path: Path
    graph_path: Path
    split: str  # "train" / "test" / "all" etc.

    def load_graph_json(self) -> dict[str, Any]:
        # Read-only, deterministic
        import json

        return json.loads(self.graph_path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Dataset:
    root: Path
    metadata: DatasetMetadata
    _samples: tuple[DatasetSample, ...]

    def samples(self, split: str | None = None) -> Iterable[DatasetSample]:
        if split is None:
            return self._samples
        return (s for s in self._samples if s.split == split)

    def splits(self) -> tuple[str, ...]:
        return tuple(sorted({s.split for s in self._samples}))
