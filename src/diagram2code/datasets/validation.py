from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DatasetError(ValueError):
    pass


# -----------------------------
# Layout + file discovery (Step 1)
# -----------------------------


@dataclass(frozen=True)
class DatasetLayout:
    root: Path
    metadata_path: Path
    images_dir: Path
    graphs_dir: Path


ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".svg"}


def assert_exists(path: Path, what: str) -> None:
    if not path.exists():
        raise DatasetError(f"Missing {what}: {path}")


def list_image_ids(images_dir: Path) -> set[str]:
    ids: set[str] = set()
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_EXTS:
            ids.add(p.stem)
    return ids


def list_graph_ids(graphs_dir: Path) -> set[str]:
    return {p.stem for p in graphs_dir.glob("*.json") if p.is_file()}


def validate_pairs(image_ids: set[str], graph_ids: set[str]) -> None:
    if image_ids != graph_ids:
        missing_graphs = sorted(image_ids - graph_ids)
        missing_images = sorted(graph_ids - image_ids)
        msg = ["Image/graph ID mismatch:"]
        if missing_graphs:
            msg.append(f"  missing graphs for: {missing_graphs[:20]}")
        if missing_images:
            msg.append(f"  missing images for: {missing_images[:20]}")
        raise DatasetError("\n".join(msg))


def validate_splits(
    all_ids: set[str],
    splits: dict[str, Iterable[str]],
) -> dict[str, tuple[str, ...]]:
    seen: set[str] = set()
    out: dict[str, tuple[str, ...]] = {}

    for split_name, ids in splits.items():
        ids_tuple = tuple(ids)
        for sid in ids_tuple:
            if sid not in all_ids:
                raise DatasetError(f"Split '{split_name}' references unknown sample_id: {sid}")
            if sid in seen:
                raise DatasetError(f"Duplicate sample_id across splits: {sid}")
            seen.add(sid)
        out[split_name] = ids_tuple

    return out


# -----------------------------
# Metadata + graph schema validation (Step 2)
# -----------------------------

SUPPORTED_DATASET_SCHEMA_VERSIONS = {"1.0"}


def validate_dataset_metadata(
    raw: dict[str, Any],
    *,
    default_name: str,
) -> tuple[str, str, str, dict[str, Any] | None, dict[str, Any]]:
    if "schema_version" not in raw:
        raise DatasetError("dataset.json missing required field: schema_version")

    schema_version = raw["schema_version"]
    if not isinstance(schema_version, str):
        raise DatasetError("dataset.json schema_version must be a string")

    if schema_version not in SUPPORTED_DATASET_SCHEMA_VERSIONS:
        raise DatasetError(
            f"Unsupported dataset schema_version: {schema_version!r}. "
            f"Supported: {sorted(SUPPORTED_DATASET_SCHEMA_VERSIONS)}"
        )

    name = raw.get("name", default_name)
    version = raw.get("version", "0.0")

    if not isinstance(name, str):
        raise DatasetError("dataset.json name must be a string")
    if not isinstance(version, str):
        raise DatasetError("dataset.json version must be a string")

    raw_splits = raw.get("splits", None)

    extra_keys = {"schema_version", "name", "version", "splits"}
    extra: dict[str, Any] = {k: v for k, v in raw.items() if k not in extra_keys}

    return schema_version, name, version, raw_splits, extra


def validate_graph_json(data: object, *, sample_id: str, path: Path) -> None:
    if not isinstance(data, dict):
        raise DatasetError(f"Graph must be a JSON object for sample_id={sample_id}: {path}")

    if "nodes" not in data or "edges" not in data:
        raise DatasetError(
            f"Graph missing required keys ('nodes', 'edges') for sample_id={sample_id}: {path}"
        )

    nodes = data["nodes"]
    edges = data["edges"]

    if not isinstance(nodes, list):
        raise DatasetError(f"'nodes' must be a list for sample_id={sample_id}: {path}")
    if not isinstance(edges, list):
        raise DatasetError(f"'edges' must be a list for sample_id={sample_id}: {path}")

    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise DatasetError(f"Node[{i}] must be an object for sample_id={sample_id}: {path}")

        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise DatasetError(
                f"Node[{i}] missing valid 'id' (non-empty string) for sample_id={sample_id}: {path}"
            )

    for i, edge in enumerate(edges):
        if not isinstance(edge, dict):
            raise DatasetError(f"Edge[{i}] must be an object for sample_id={sample_id}: {path}")

        source = edge.get("source")
        target = edge.get("target")

        if not isinstance(source, str) or not source:
            raise DatasetError(
                f"Edge[{i}] missing valid 'source' (non-empty string) "
                f"for sample_id={sample_id}: {path}"
            )
        if not isinstance(target, str) or not target:
            raise DatasetError(
                f"Edge[{i}] missing valid 'target' (non-empty string) "
                f"for sample_id={sample_id}: {path}"
            )


def load_and_validate_graph(path: Path, *, sample_id: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"Invalid JSON in graph for sample_id={sample_id}: {path}") from exc

    validate_graph_json(raw, sample_id=sample_id, path=path)
    return raw  # validated to be dict
