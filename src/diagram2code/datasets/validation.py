from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class DatasetLayout:
    root: Path
    metadata_path: Path
    images_dir: Path
    graphs_dir: Path


ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


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
