from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    id: int
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
