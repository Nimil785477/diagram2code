from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class _NodeSpec:
    id: int
    x: int
    y: int
    w: int
    h: int

    def bbox(self) -> list[int]:
        return [self.x, self.y, self.w, self.h]

    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


def _draw_rect(img: np.ndarray, n: _NodeSpec, *, thickness: int = 6) -> None:
    # White fill with thick black outline (matches your fixture style)
    cv2.rectangle(img, (n.x, n.y), (n.x + n.w, n.y + n.h), (255, 255, 255), -1)
    cv2.rectangle(img, (n.x, n.y), (n.x + n.w, n.y + n.h), (0, 0, 0), thickness)


def _draw_arrow(img: np.ndarray, a: _NodeSpec, b: _NodeSpec, *, thickness: int = 6) -> None:
    ax, ay = a.center()
    bx, by = b.center()

    # Trim so arrow doesn't start deep inside the rectangles
    dx, dy = bx - ax, by - ay
    dist = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / dist, dy / dist
    trim = max(a.w, a.h) * 0.25

    start = (int(ax + ux * trim), int(ay + uy * trim))
    end = (int(bx - ux * trim), int(by - uy * trim))

    cv2.arrowedLine(img, start, end, (0, 0, 0), thickness, tipLength=0.25)


def _write_sample(
    *,
    images_dir: Path,
    graphs_dir: Path,
    stem: str,
    size: tuple[int, int],
    nodes: list[_NodeSpec],
    edges: list[tuple[int, int]],
) -> None:
    w, h = size
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    for n in nodes:
        _draw_rect(img, n)

    id_to_node = {n.id: n for n in nodes}
    for u, v in edges:
        _draw_arrow(img, id_to_node[u], id_to_node[v])

    img_path = images_dir / f"{stem}.png"
    ok = cv2.imwrite(str(img_path), img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {img_path}")

    graph = {
        "nodes": [{"id": n.id, "bbox": n.bbox()} for n in nodes],
        "edges": [{"from": u, "to": v} for (u, v) in edges],
    }
    (graphs_dir / f"{stem}.graph.json").write_text(json.dumps(graph), encoding="utf-8")


def generate_synthetic_basic(root: Path, *, n: int = 3) -> Path:
    """
    Generate a simple legacy synthetic dataset under:
      root/
        images/
        graphs/

    Notes:
    - Node ids are ints (legacy).
    - Bboxes are [x,y,w,h].
    - Graph files are named <stem>.graph.json (legacy).
    - This is kept for backward compatibility with older benchmark code.
    """
    root.mkdir(parents=True, exist_ok=True)
    images_dir = root / "images"
    graphs_dir = root / "graphs"
    images_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic, small, stable patterns (good for tests).
    # Use at least 1 sample even if n=0.
    n = max(1, n)

    patterns: list[tuple[str, tuple[int, int], list[_NodeSpec], list[tuple[int, int]]]] = []

    # 1) chain: 0 -> 1
    patterns.append(
        (
            "chain",
            (400, 300),
            [
                _NodeSpec(0, 50, 100, 80, 80),
                _NodeSpec(1, 240, 100, 80, 80),
            ],
            [(0, 1)],
        )
    )

    # 2) branching: 0 -> 1 and 0 -> 2
    patterns.append(
        (
            "branch",
            (500, 350),
            [
                _NodeSpec(0, 60, 120, 80, 80),
                _NodeSpec(1, 260, 60, 80, 80),
                _NodeSpec(2, 260, 200, 80, 80),
            ],
            [(0, 1), (0, 2)],
        )
    )

    # 3) diamond: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
    patterns.append(
        (
            "diamond",
            (600, 400),
            [
                _NodeSpec(0, 60, 160, 80, 80),
                _NodeSpec(1, 260, 80, 80, 80),
                _NodeSpec(2, 260, 240, 80, 80),
                _NodeSpec(3, 460, 160, 80, 80),
            ],
            [(0, 1), (0, 2), (1, 3), (2, 3)],
        )
    )

    for i in range(n):
        name, size, nodes, edges = patterns[i % len(patterns)]
        _write_sample(
            images_dir=images_dir,
            graphs_dir=graphs_dir,
            stem=f"{i:03d}_{name}",
            size=size,
            nodes=nodes,
            edges=edges,
        )

    return root
