from __future__ import annotations

from pathlib import Path

import pytest

from diagram2code.datasets.loader import load_dataset
from diagram2code.datasets.validation import DatasetError


def test_load_dataset_rejects_raw_only_install(tmp_path: Path) -> None:
    root = tmp_path / "flowlearn_raw_only"
    (root / "raw" / "FlowLearn").mkdir(parents=True)

    with pytest.raises(DatasetError) as ei:
        load_dataset(root)

    msg = str(ei.value).lower()
    assert "raw-only" in msg or "raw only" in msg
    assert "dataset.json" in msg
