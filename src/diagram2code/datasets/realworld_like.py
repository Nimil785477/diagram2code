from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


@dataclass(frozen=True)
class _Node:
    id: str
    bbox: list[int]  # [x, y, w, h]


@dataclass(frozen=True)
class _Edge:
    source: str
    target: str


_BG = "white"
_FG = "black"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _center(bbox: list[int]) -> tuple[int, int]:
    x, y, w, h = bbox
    return (x + w // 2, y + h // 2)


def _clamp_bbox(x: int, y: int, w: int, h: int, canvas_w: int, canvas_h: int) -> list[int]:
    x = max(10, min(canvas_w - w - 10, x))
    y = max(10, min(canvas_h - h - 10, y))
    return [x, y, w, h]


def _jitter(v: int, amount: int, rng: random.Random) -> int:
    return v + rng.randint(-amount, amount)


def _make_bbox(
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    canvas_w: int,
    canvas_h: int,
    rng: random.Random,
    pos_jitter: int = 10,
    size_jitter_w: int = 10,
    size_jitter_h: int = 6,
) -> list[int]:
    jw = max(60, _jitter(w, size_jitter_w, rng))
    jh = max(35, _jitter(h, size_jitter_h, rng))
    jx = _jitter(x, pos_jitter, rng)
    jy = _jitter(y, pos_jitter, rng)
    return _clamp_bbox(jx, jy, jw, jh, canvas_w, canvas_h)


def _arrow_head(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    size: int = 10,
) -> None:
    angle = math.atan2(y1 - y0, x1 - x0)
    left = (
        int(x1 - size * math.cos(angle - math.pi / 6)),
        int(y1 - size * math.sin(angle - math.pi / 6)),
    )
    right = (
        int(x1 - size * math.cos(angle + math.pi / 6)),
        int(y1 - size * math.sin(angle + math.pi / 6)),
    )
    draw.polygon([(x1, y1), left, right], fill=_FG)


def _draw_edge(
    draw: ImageDraw.ImageDraw,
    src: _Node,
    dst: _Node,
    *,
    line_width: int = 3,
    trim: int = 18,
) -> None:
    sx, sy = _center(src.bbox)
    dx, dy = _center(dst.bbox)

    vx = dx - sx
    vy = dy - sy
    dist = math.hypot(vx, vy) or 1.0
    ux = vx / dist
    uy = vy / dist

    start = (int(sx + ux * trim), int(sy + uy * trim))
    end = (int(dx - ux * trim), int(dy - uy * trim))

    draw.line([start, end], fill=_FG, width=line_width)
    _arrow_head(draw, start[0], start[1], end[0], end[1], size=10)


def _draw_node(draw: ImageDraw.ImageDraw, node: _Node, *, line_width: int = 3) -> None:
    x, y, w, h = node.bbox
    draw.rectangle([x, y, x + w, y + h], outline=_FG, width=line_width)


def _render_sample(
    *,
    image_path: Path,
    canvas_size: tuple[int, int],
    nodes: list[_Node],
    edges: list[_Edge],
) -> None:
    canvas_w, canvas_h = canvas_size
    img = Image.new("RGB", (canvas_w, canvas_h), color=_BG)
    draw = ImageDraw.Draw(img)

    node_by_id = {n.id: n for n in nodes}

    for edge in edges:
        _draw_edge(draw, node_by_id[edge.source], node_by_id[edge.target])

    for node in nodes:
        _draw_node(draw, node)

    img.save(image_path)


def _write_graph(graph_path: Path, nodes: list[_Node], edges: list[_Edge]) -> None:
    payload = {
        "nodes": [{"id": n.id, "bbox": n.bbox} for n in nodes],
        "edges": [{"source": e.source, "target": e.target} for e in edges],
    }
    graph_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _motif_simple_horizontal(
    rng: random.Random,
) -> tuple[tuple[int, int], list[_Node], list[_Edge]]:
    canvas = (360, 320)
    nodes = [
        _Node("A", _make_bbox(x=50, y=120, w=90, h=70, canvas_w=360, canvas_h=320, rng=rng)),
        _Node("B", _make_bbox(x=215, y=120, w=88, h=68, canvas_w=360, canvas_h=320, rng=rng)),
    ]
    edges = [_Edge("A", "B")]
    return canvas, nodes, edges


def _motif_branch_merge(rng: random.Random) -> tuple[tuple[int, int], list[_Node], list[_Edge]]:
    canvas = (576, 324)
    nodes = [
        _Node("A", _make_bbox(x=40, y=120, w=82, h=78, canvas_w=576, canvas_h=324, rng=rng)),
        _Node("B", _make_bbox(x=210, y=38, w=82, h=78, canvas_w=576, canvas_h=324, rng=rng)),
        _Node("C", _make_bbox(x=210, y=205, w=82, h=78, canvas_w=576, canvas_h=324, rng=rng)),
        _Node("D", _make_bbox(x=405, y=120, w=82, h=78, canvas_w=576, canvas_h=324, rng=rng)),
    ]
    edges = [_Edge("A", "B"), _Edge("A", "C"), _Edge("B", "D"), _Edge("C", "D")]
    return canvas, nodes, edges


def _motif_staged_directional(
    rng: random.Random,
) -> tuple[tuple[int, int], list[_Node], list[_Edge]]:
    canvas = (400, 300)
    nodes = [
        _Node("A", _make_bbox(x=70, y=40, w=78, h=60, canvas_w=400, canvas_h=300, rng=rng)),
        _Node("B", _make_bbox(x=225, y=45, w=78, h=60, canvas_w=400, canvas_h=300, rng=rng)),
        _Node("C", _make_bbox(x=38, y=185, w=78, h=60, canvas_w=400, canvas_h=300, rng=rng)),
        _Node("D", _make_bbox(x=225, y=175, w=78, h=60, canvas_w=400, canvas_h=300, rng=rng)),
    ]
    edges = [_Edge("C", "A"), _Edge("A", "B"), _Edge("C", "D"), _Edge("B", "D")]
    return canvas, nodes, edges


def _motif_library():
    return [
        _motif_simple_horizontal,
        _motif_branch_merge,
        _motif_staged_directional,
    ]


def build_realworld_like_dataset(
    *,
    out: Path,
    split: str = "test",
    num_samples: int = 12,
    seed: int = 0,
) -> Path:
    """
    Build a small deterministic Phase-3 dataset from motif-driven layouts.

    Output layout:
      out/
        dataset.json
        splits.json
        images/<sample_id>.png
        graphs/<sample_id>.json
    """
    if split not in {"train", "test", "all"}:
        raise ValueError("split must be one of: train, test, all")
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")

    out = Path(out)
    images_dir = out / "images"
    graphs_dir = out / "graphs"
    _ensure_dir(out)
    _ensure_dir(images_dir)
    _ensure_dir(graphs_dir)

    rng = random.Random(seed)
    motifs = _motif_library()

    sample_ids: list[str] = []

    for i in range(num_samples):
        sample_id = f"realworld-like-{i:04d}"
        sample_ids.append(sample_id)

        motif = rng.choice(motifs)
        canvas_size, nodes, edges = motif(rng)

        _render_sample(
            image_path=images_dir / f"{sample_id}.png",
            canvas_size=canvas_size,
            nodes=nodes,
            edges=edges,
        )
        _write_graph(graphs_dir / f"{sample_id}.json", nodes, edges)

    dataset_json: dict[str, Any] = {
        "schema_version": "1.0",
        "name": "realworld-like-v1",
        "version": "0.1",
        "splits": {split: sample_ids},
        "generator": {
            "type": "realworld_like",
            "num_samples": num_samples,
            "seed": seed,
            "motifs": [
                "simple_horizontal",
                "branch_merge",
                "staged_directional",
            ],
        },
    }
    splits_json: dict[str, Any] = {
        "schema_version": "1.0",
        "splits": {split: sample_ids},
    }

    (out / "dataset.json").write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")
    (out / "splits.json").write_text(json.dumps(splits_json, indent=2), encoding="utf-8")

    return out
