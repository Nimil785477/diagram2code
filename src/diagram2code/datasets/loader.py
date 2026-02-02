from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import Dataset, DatasetMetadata, DatasetSample
from .validation import (
    ALLOWED_IMAGE_EXTS,
    DatasetError,
    DatasetLayout,
    assert_exists,
    list_graph_ids,
    list_image_ids,
    validate_pairs,
    validate_splits,
)


def _pick_image_path(images_dir: Path, sample_id: str) -> Path:
    # deterministic selection order by extension
    for ext in sorted(ALLOWED_IMAGE_EXTS):
        p = images_dir / f"{sample_id}{ext}"
        if p.exists():
            return p
    raise DatasetError(f"No image found for sample_id={sample_id} in {images_dir}")


def load_dataset(root: str | Path) -> Dataset:
    root = Path(root)
    layout = DatasetLayout(
        root=root,
        metadata_path=root / "dataset.json",
        images_dir=root / "images",
        graphs_dir=root / "graphs",
    )

    assert_exists(layout.metadata_path, "dataset.json")
    assert_exists(layout.images_dir, "images/ directory")
    assert_exists(layout.graphs_dir, "graphs/ directory")

    raw = json.loads(layout.metadata_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise DatasetError("dataset.json must be a JSON object")

    schema_version = str(raw.get("schema_version", "1.0"))
    name = str(raw.get("name", root.name))
    version = str(raw.get("version", "0.0"))
    raw_splits = raw.get("splits", None)

    image_ids = list_image_ids(layout.images_dir)
    graph_ids = list_graph_ids(layout.graphs_dir)
    validate_pairs(image_ids, graph_ids)

    extra_keys = {"schema_version", "name", "version", "splits"}
    extra: dict[str, Any] = {k: v for k, v in raw.items() if k not in extra_keys}

    splits: dict[str, tuple[str, ...]] = {}
    sample_split: dict[str, str] = {}

    if raw_splits is None:
        splits = {"all": tuple(sorted(image_ids))}
        for sid in splits["all"]:
            sample_split[sid] = "all"
    else:
        if not isinstance(raw_splits, dict):
            raise DatasetError("dataset.json 'splits' must be an object: {split_name: [ids...]}")
        normalized = {str(k): v for k, v in raw_splits.items()}
        validated = validate_splits(image_ids, normalized)  # also checks duplicates/unknown IDs
        splits = validated
        for split_name, ids in splits.items():
            for sid in ids:
                sample_split[sid] = split_name

        # Optional strictness: ensure all ids are assigned to some split
        unassigned = sorted(image_ids - set(sample_split.keys()))
        if unassigned:
            raise DatasetError(f"Unassigned sample_ids (not in any split): {unassigned[:20]}")

    metadata = DatasetMetadata(
        schema_version=schema_version,
        name=name,
        version=version,
        splits={k: tuple(v) for k, v in splits.items()},
        extra=extra,
    )

    samples: list[DatasetSample] = []
    for sid in sorted(image_ids):
        samples.append(
            DatasetSample(
                sample_id=sid,
                image_path=_pick_image_path(layout.images_dir, sid),
                graph_path=layout.graphs_dir / f"{sid}.json",
                split=sample_split[sid],
            )
        )

    return Dataset(root=root, metadata=metadata, _samples=tuple(samples))
