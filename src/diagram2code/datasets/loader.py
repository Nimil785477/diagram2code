from __future__ import annotations

import json
from pathlib import Path

from .types import Dataset, DatasetMetadata, DatasetSample
from .validation import (
    ALLOWED_IMAGE_EXTS,
    DatasetError,
    DatasetLayout,
    assert_exists,
    list_graph_ids,
    list_image_ids,
    load_and_validate_graph,
    validate_dataset_metadata,
    validate_pairs,
    validate_splits,
)


def _pick_image_path(images_dir: Path, sample_id: str) -> Path:
    for ext in sorted(ALLOWED_IMAGE_EXTS):
        candidate = images_dir / f"{sample_id}{ext}"
        if candidate.exists():
            return candidate

    raise DatasetError(f"No image found for sample_id={sample_id} in {images_dir}")


def load_dataset(
    root: str | Path,
    *,
    validate_graphs: bool = True,
) -> Dataset:
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

    (
        schema_version,
        name,
        version,
        raw_splits,
        extra,
    ) = validate_dataset_metadata(raw, default_name=root.name)

    image_ids = list_image_ids(layout.images_dir)
    graph_ids = list_graph_ids(layout.graphs_dir)
    validate_pairs(image_ids, graph_ids)

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
        validated = validate_splits(image_ids, normalized)
        splits = validated

        for split_name, ids in splits.items():
            for sid in ids:
                sample_split[sid] = split_name

        unassigned = sorted(image_ids - set(sample_split.keys()))
        if unassigned:
            raise DatasetError(f"Unassigned sample_ids (not in any split): {unassigned[:20]}")

    samples: list[DatasetSample] = []

    for sid in sorted(image_ids):
        image_path = _pick_image_path(layout.images_dir, sid)
        graph_path = layout.graphs_dir / f"{sid}.json"

        if validate_graphs:
            load_and_validate_graph(graph_path, sample_id=sid)

        samples.append(
            DatasetSample(
                sample_id=sid,
                image_path=image_path,
                graph_path=graph_path,
                split=sample_split[sid],
            )
        )

    metadata = DatasetMetadata(
        schema_version=schema_version,
        name=name,
        version=version,
        splits={k: tuple(v) for k, v in splits.items()},
        extra=extra,
    )

    return Dataset(
        root=root,
        metadata=metadata,
        _samples=tuple(samples),
    )
