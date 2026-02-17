from __future__ import annotations

import json
from pathlib import Path

from diagram2code.datasets.layout import DatasetLayout
from diagram2code.datasets.types import Dataset
from diagram2code.datasets.validation import DatasetError, assert_exists


def _all_samples_under_images_dir(data: dict) -> bool:
    """
    Backward-compat validation:
    - Old datasets store image_path like 'images/xyz.png' and expect root/images/ to exist.
    - Newer/converted datasets (e.g. FlowLearn HF snapshot) may store image_path elsewhere
      like 'raw/FlowLearn/...'.

    Rule:
      If *all* samples are under 'images/', require layout.images_dir exists.
      Otherwise, don't require images/ at root.
    """
    samples = data.get("samples") or []
    if not samples:
        return False

    def _is_under_images(p: str) -> bool:
        p2 = Path(p).as_posix().lstrip("./")
        return p2 == "images" or p2.startswith("images/")

    image_paths = []
    for s in samples:
        ip = s.get("image_path")
        if isinstance(ip, str) and ip:
            image_paths.append(ip)

    if not image_paths:
        return False

    return all(_is_under_images(ip) for ip in image_paths)


def load_dataset(dataset_root: str | Path, *, validate_graphs: bool = True) -> Dataset:
    root = Path(dataset_root)
    layout = DatasetLayout(root)

    assert_exists(layout.metadata_path, "dataset.json")

    try:
        data = json.loads(layout.metadata_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise DatasetError(f"Failed to parse dataset.json: {layout.metadata_path}") from e

    # Only require images/ if the dataset schema is using images/... paths.
    if _all_samples_under_images_dir(data):
        assert_exists(layout.images_dir, "images/ directory")

    return Dataset.from_json_dict(root=root, data=data, validate_graphs=validate_graphs)
