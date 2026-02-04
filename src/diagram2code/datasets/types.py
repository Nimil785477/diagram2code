from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    schema_version: str
    name: str
    version: str
    splits: dict[str, tuple[str, ...]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DatasetSample:
    sample_id: str
    image_path: Path
    graph_path: Path
    split: str = "all"

    def load_graph_json(self) -> dict[str, Any]:
        raw = json.loads(self.graph_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Graph JSON must be an object: {self.graph_path}")
        return raw


@dataclass(frozen=True, slots=True)
class Dataset:
    root: Path
    metadata: DatasetMetadata
    _samples: tuple[DatasetSample, ...] = ()

    def splits(self) -> tuple[str, ...]:
        return tuple(sorted(self.metadata.splits.keys()))

    def samples(self, split: str | None = None) -> Iterable[DatasetSample]:
        if split is None:
            return self._samples
        return tuple(s for s in self._samples if s.split == split)

    @property
    def items(self) -> tuple[DatasetSample, ...]:
        return self._samples

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self):
        return iter(self._samples)
