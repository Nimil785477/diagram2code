from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class _Node:
    id: str
    label: str
    bbox: list[int]  # [x, y, w, h]


@dataclass(frozen=True)
class _Edge:
    source: str
    target: str


_CANVAS_W = 640
_CANVAS_H = 360
_NODE_W = 120
_NODE_H = 56
_BG = "white"
_FG = "black"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _center(x: int, y: int, w: int, h: int) -> tuple[int, int]:
    return (x + w // 2, y + h // 2)


def _arrow_head(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> None:
    angle = math.atan2(y1 - y0, x1 - x0)
    size = 10
    left = (
        int(x1 - size * math.cos(angle - math.pi / 6)),
        int(y1 - size * math.sin(angle - math.pi / 6)),
    )
    right = (
        int(x1 - size * math.cos(angle + math.pi / 6)),
        int(y1 - size * math.sin(angle + math.pi / 6)),
    )
    draw.polygon([(x1, y1), left, right], fill=_FG)


def _draw_edge(draw: ImageDraw.ImageDraw, src: _Node, dst: _Node) -> None:
    sx, sy = _center(*src.bbox)
    dx, dy = _center(*dst.bbox)

    # Simple routing:
    # horizontal if mostly same row, otherwise vertical-ish direct line.
    draw.line([(sx, sy), (dx, dy)], fill=_FG, width=3)
    _arrow_head(draw, sx, sy, dx, dy)


def _draw_node(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, node: _Node) -> None:
    x, y, w, h = node.bbox
    draw.rectangle([x, y, x + w, y + h], outline=_FG, width=3)

    # crude text centering
    text_bbox = draw.textbbox((0, 0), node.label, font=font)
    tw = text_bbox[2] - text_bbox[0]
    th = text_bbox[3] - text_bbox[1]
    tx = x + (w - tw) // 2
    ty = y + (h - th) // 2
    draw.text((tx, ty), node.label, fill=_FG, font=font)


def _pattern_chain() -> tuple[list[_Node], list[_Edge]]:
    nodes = [
        _Node("0", "Start", [60, 150, _NODE_W, _NODE_H]),
        _Node("1", "Process", [260, 150, _NODE_W, _NODE_H]),
        _Node("2", "End", [460, 150, _NODE_W, _NODE_H]),
    ]
    edges = [_Edge("0", "1"), _Edge("1", "2")]
    return nodes, edges


def _pattern_branch() -> tuple[list[_Node], list[_Edge]]:
    nodes = [
        _Node("0", "Start", [60, 150, _NODE_W, _NODE_H]),
        _Node("1", "Check", [240, 60, _NODE_W, _NODE_H]),
        _Node("2", "Yes", [440, 40, _NODE_W, _NODE_H]),
        _Node("3", "No", [440, 220, _NODE_W, _NODE_H]),
    ]
    edges = [_Edge("0", "1"), _Edge("1", "2"), _Edge("1", "3")]
    return nodes, edges


def _pattern_merge() -> tuple[list[_Node], list[_Edge]]:
    nodes = [
        _Node("0", "Input", [60, 150, _NODE_W, _NODE_H]),
        _Node("1", "Left", [240, 70, _NODE_W, _NODE_H]),
        _Node("2", "Right", [240, 230, _NODE_W, _NODE_H]),
        _Node("3", "Merge", [460, 150, _NODE_W, _NODE_H]),
    ]
    edges = [_Edge("0", "1"), _Edge("0", "2"), _Edge("1", "3"), _Edge("2", "3")]
    return nodes, edges


def _pattern_library() -> list[tuple[list[_Node], list[_Edge]]]:
    return [
        _pattern_chain(),
        _pattern_branch(),
        _pattern_merge(),
    ]


def _render_sample(image_path: Path, nodes: list[_Node], edges: list[_Edge]) -> None:
    img = Image.new("RGB", (_CANVAS_W, _CANVAS_H), color=_BG)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    node_by_id = {n.id: n for n in nodes}

    for edge in edges:
        _draw_edge(draw, node_by_id[edge.source], node_by_id[edge.target])

    for node in nodes:
        _draw_node(draw, font, node)

    img.save(image_path)


def _write_graph(graph_path: Path, nodes: list[_Node], edges: list[_Edge]) -> None:
    payload = {
        "nodes": [{"id": n.id, "bbox": n.bbox} for n in nodes],
        "edges": [{"source": e.source, "target": e.target} for e in edges],
    }
    graph_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_synthflow_dataset(
    *,
    out: Path,
    split: str = "test",
    num_samples: int = 20,
    seed: int = 0,
) -> Path:
    """
    Build a small deterministic Phase-3 benchmark dataset.

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
    patterns = _pattern_library()

    sample_ids: list[str] = []

    for i in range(num_samples):
        sample_id = f"synthflow-{i:04d}"
        sample_ids.append(sample_id)

        nodes, edges = rng.choice(patterns)

        image_path = images_dir / f"{sample_id}.png"
        graph_path = graphs_dir / f"{sample_id}.json"

        _render_sample(image_path, nodes, edges)
        _write_graph(graph_path, nodes, edges)

    dataset_json: dict[str, Any] = {
        "schema_version": "1.0",
        "name": "synthflow_v1",
        "version": "1",
        "splits": {split: sample_ids},
        "extra": {
            "generator": "synthflow_v1",
            "num_samples": num_samples,
            "seed": seed,
        },
    }
    splits_json: dict[str, Any] = {
        "schema_version": "1.0",
        "splits": {split: sample_ids},
    }

    (out / "dataset.json").write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")
    (out / "splits.json").write_text(json.dumps(splits_json, indent=2), encoding="utf-8")

    return out
