from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DatasetValidationError(ValueError):
    """Raised when a dataset folder or an item is invalid."""


@dataclass(frozen=True)
class GTNode:
    id: int
    bbox: tuple[int, int, int, int]  # (x, y, w, h)


@dataclass(frozen=True)
class GTEdge:
    from_id: int
    to_id: int


@dataclass(frozen=True)
class GTGraph:
    nodes: list[GTNode]
    edges: list[GTEdge]


@dataclass(frozen=True)
class DatasetItem:
    image_path: Path
    graph_path: Path
    gt: GTGraph


@dataclass(frozen=True)
class Dataset:
    root: Path
    name: str
    items: list[DatasetItem]


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in {".png", ".jpg", ".jpeg"}


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        # allow UTF-8 BOM files
        return json.loads(path.read_text(encoding="utf-8-sig"))


def _parse_bbox(b: Any) -> tuple[int, int, int, int]:
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        raise DatasetValidationError(f"Invalid bbox (expected 4 ints): {b!r}")
    x, y, w, h = b
    if not all(isinstance(v, int) for v in (x, y, w, h)):
        raise DatasetValidationError(f"Invalid bbox values (must be ints): {b!r}")
    if w <= 0 or h <= 0:
        raise DatasetValidationError(f"Invalid bbox size (w,h must be >0): {b!r}")
    return x, y, w, h


def _parse_gt_graph(data: Any, *, graph_path: Path) -> GTGraph:
    if not isinstance(data, dict):
        raise DatasetValidationError(f"{graph_path}: GT graph must be an object")

    nodes_raw = data.get("nodes")
    edges_raw = data.get("edges")

    if not isinstance(nodes_raw, list):
        raise DatasetValidationError(f"{graph_path}: 'nodes' must be a list")
    if not isinstance(edges_raw, list):
        raise DatasetValidationError(f"{graph_path}: 'edges' must be a list")

    nodes: list[GTNode] = []
    seen_ids: set[int] = set()
    for n in nodes_raw:
        if not isinstance(n, dict):
            raise DatasetValidationError(f"{graph_path}: node must be an object: {n!r}")
        if "id" not in n:
            raise DatasetValidationError(f"{graph_path}: node missing 'id': {n!r}")
        if "bbox" not in n:
            raise DatasetValidationError(f"{graph_path}: node missing 'bbox': {n!r}")

        node_id = n["id"]
        if not isinstance(node_id, int):
            raise DatasetValidationError(f"{graph_path}: node id must be int: {node_id!r}")
        if node_id in seen_ids:
            raise DatasetValidationError(f"{graph_path}: duplicate node id: {node_id}")
        seen_ids.add(node_id)

        bbox = _parse_bbox(n["bbox"])
        nodes.append(GTNode(id=node_id, bbox=bbox))

    edges: list[GTEdge] = []
    for e in edges_raw:
        if not isinstance(e, dict):
            raise DatasetValidationError(f"{graph_path}: edge must be an object: {e!r}")
        if "from" not in e or "to" not in e:
            raise DatasetValidationError(f"{graph_path}: edge missing 'from'/'to': {e!r}")
        u = e["from"]
        v = e["to"]
        if not isinstance(u, int) or not isinstance(v, int):
            raise DatasetValidationError(f"{graph_path}: edge endpoints must be ints: {e!r}")
        if u not in seen_ids or v not in seen_ids:
            raise DatasetValidationError(f"{graph_path}: edge references unknown node: {e!r}")
        if u == v:
            raise DatasetValidationError(f"{graph_path}: self-loop not allowed: {e!r}")
        edges.append(GTEdge(from_id=u, to_id=v))

    return GTGraph(nodes=nodes, edges=edges)


def _validate_bboxes_with_image(gt: GTGraph, image_path: Path) -> None:
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        raise DatasetValidationError(f"Could not read image: {image_path}")
    h, w = img.shape[:2]

    for n in gt.nodes:
        x, y, bw, bh = n.bbox
        if x < 0 or y < 0:
            raise DatasetValidationError(f"{image_path}: bbox out of bounds (neg): {n.bbox}")
        if x + bw > w or y + bh > h:
            raise DatasetValidationError(
                f"{image_path}: bbox out of bounds: {n.bbox} (img {w}x{h})"
            )


def load_dataset(dataset_root: str | Path, *, validate_bboxes: bool = True) -> Dataset:
    """
    Load a dataset in the frozen contract:

      <root>/
        images/
        graphs/
        README.md (optional but recommended)

    Pairs: images/<stem>.(png|jpg|jpeg) with graphs/<stem>.graph.json
    """
    root = Path(dataset_root)
    if not root.exists() or not root.is_dir():
        raise DatasetValidationError(f"Dataset root not found or not a directory: {root}")

    images_dir = root / "images"
    graphs_dir = root / "graphs"
    if not images_dir.is_dir():
        raise DatasetValidationError(f"Missing images/ directory: {images_dir}")
    if not graphs_dir.is_dir():
        raise DatasetValidationError(f"Missing graphs/ directory: {graphs_dir}")

    images = sorted([p for p in images_dir.iterdir() if p.is_file() and _is_image(p)])
    if not images:
        raise DatasetValidationError(f"No images found in: {images_dir}")

    items: list[DatasetItem] = []
    for img_path in images:
        stem = img_path.stem
        graph_path = graphs_dir / f"{stem}.graph.json"
        if not graph_path.exists():
            raise DatasetValidationError(
                f"Missing graph for image {img_path.name}: {graph_path.name}"
            )

        data = _load_json(graph_path)
        gt = _parse_gt_graph(data, graph_path=graph_path)

        if validate_bboxes:
            _validate_bboxes_with_image(gt, img_path)

        items.append(DatasetItem(image_path=img_path, graph_path=graph_path, gt=gt))

    return Dataset(root=root, name=root.name, items=items)
