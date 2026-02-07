from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class BenchmarkResult:
    # Frozen, stable top-level contract
    schema_version: str
    dataset: str
    split: str
    predictor: str
    num_samples: int
    metrics: Mapping[str, float]

    # Optional, but reserved for reproducibility/debug
    run: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version={self.schema_version!r}")
        if self.num_samples < 0:
            raise ValueError("num_samples must be >= 0")
