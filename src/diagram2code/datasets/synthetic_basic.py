from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class _NodeSpec:
    id: str
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

    # Trim so arrow doesn't start deep inside rectangles
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
    sample_id: str,
    size: tuple[int, int],
    nodes: list[_NodeSpec],
    edges: list[tuple[str, str]],
) -> None:
    w, h = size
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    for n in nodes:
        _draw_rect(img, n)

    id_to_node = {n.id: n for n in nodes}
    for u, v in edges:
        _draw_arrow(img, id_to_node[u], id_to_node[v])

    img_path = images_dir / f"{sample_id}.png"
    ok = cv2.imwrite(str(img_path), img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {img_path}")

    # Phase 3 graph schema (Step 2):
    # nodes: list of objects with string "id"
    # edges: list of objects with string "source"/"target"
    # extra keys allowed (bbox)
    graph = {
        "nodes": [{"id": n.id, "bbox": n.bbox()} for n in nodes],
        "edges": [{"source": u, "target": v} for (u, v) in edges],
    }
    (graphs_dir / f"{sample_id}.json").write_text(
        json.dumps(graph, indent=2),
        encoding="utf-8",
    )


def generate_synthetic_basic(
    root: Path,
    *,
    n: int = 3,
    seed: int = 0,
    split: str = "test",
    id_width: int = 4,
) -> Path:
    """
    Generate a Phase 3 dataset-first synthetic dataset:

      root/
        dataset.json
        images/<sample_id>.png
        graphs/<sample_id>.json

    Deterministic given (n, seed, split, id_width).
    """
    root.mkdir(parents=True, exist_ok=True)
    images_dir = root / "images"
    graphs_dir = root / "graphs"
    images_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    if n <= 0:
        raise ValueError("n must be > 0")
    if not split:
        raise ValueError("split must be non-empty")

    rng = random.Random(seed)

    patterns: list[tuple[str, tuple[int, int], list[_NodeSpec], list[tuple[str, str]]]] = []

    # 1) chain: A -> B
    patterns.append(
        (
            "chain",
            (400, 300),
            [
                _NodeSpec("A", 50, 100, 80, 80),
                _NodeSpec("B", 240, 100, 80, 80),
            ],
            [("A", "B")],
        )
    )

    # 2) branching: A -> B and A -> C
    patterns.append(
        (
            "branch",
            (500, 350),
            [
                _NodeSpec("A", 60, 120, 80, 80),
                _NodeSpec("B", 260, 60, 80, 80),
                _NodeSpec("C", 260, 200, 80, 80),
            ],
            [("A", "B"), ("A", "C")],
        )
    )

    # 3) diamond: A -> B, A -> C, B -> D, C -> D
    patterns.append(
        (
            "diamond",
            (600, 400),
            [
                _NodeSpec("A", 60, 160, 80, 80),
                _NodeSpec("B", 260, 80, 80, 80),
                _NodeSpec("C", 260, 240, 80, 80),
                _NodeSpec("D", 460, 160, 80, 80),
            ],
            [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")],
        )
    )

    sample_ids: list[str] = []

    for i in range(1, n + 1):
        name, size, base_nodes, edges = patterns[(i - 1) % len(patterns)]

        jitter_x = rng.randint(-10, 10)
        jitter_y = rng.randint(-10, 10)

        nodes = [_NodeSpec(n.id, n.x + jitter_x, n.y + jitter_y, n.w, n.h) for n in base_nodes]

        sample_id = f"{str(i).zfill(id_width)}_{name}"
        sample_ids.append(sample_id)

        _write_sample(
            images_dir=images_dir,
            graphs_dir=graphs_dir,
            sample_id=sample_id,
            size=size,
            nodes=nodes,
            edges=edges,
        )

    dataset_meta = {
        "schema_version": "1.0",
        "name": "synthetic-basic",
        "version": "0.1",
        "splits": {split: sample_ids},
        "generator": {
            "type": "synthetic_basic",
            "n": n,
            "seed": seed,
            "id_width": id_width,
        },
    }
    (root / "dataset.json").write_text(
        json.dumps(dataset_meta, indent=2),
        encoding="utf-8",
    )

    return root
