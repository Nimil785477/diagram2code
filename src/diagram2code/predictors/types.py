from __future__ import annotations

from typing import NotRequired, TypedDict


class PredNode(TypedDict):
    id: str
    bbox: list[float]  # [x, y, w, h] in image pixel space
    label: NotRequired[str]


class PredEdge(TypedDict):
    source: str
    target: str
    direction: NotRequired[str]


class GraphPrediction(TypedDict):
    nodes: list[PredNode]
    edges: list[PredEdge]
