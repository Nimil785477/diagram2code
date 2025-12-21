from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Node:
    id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
