from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class PredGraph:
    nodes: list[dict]  # each: {"id": ..., "bbox": [x,y,w,h]}
    edges: list[dict]  # each: {"from": ..., "to": ...}


class Predictor(Protocol):
    def __call__(self, image_path: Path) -> PredGraph: ...
