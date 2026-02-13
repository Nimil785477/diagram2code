from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from diagram2code.datasets.fetching.manifest import MANIFEST_SCHEMA_VERSION


def _make_minimal_dataset_json(ds_root: Path) -> None:
    # Phase-3 dataset loader expects dataset.json to exist
    (ds_root / "dataset.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "name": "ds",
                "version": "0.1",
                "splits": {"test": ["sample-1"]},
            }
        ),
        encoding="utf-8",
    )


def _make_minimal_manifest_json(ds_root: Path) -> None:
    # Must match DatasetManifestV1 contract
    (ds_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "name": "ds",
                "version": "0.1",
                "fetched_at_utc": "2026-02-08T00:00:00Z",
                "artifacts": [],
                "tooling": {"diagram2code_version": "0.1.6", "python": "3.x"},
            }
        ),
        encoding="utf-8",
    )


def test_cli_benchmark_fail_on_missing_manifest_errors(tmp_path: Path) -> None:
    ds_root = tmp_path / "ds"
    ds_root.mkdir()
    _make_minimal_dataset_json(ds_root)
    # NOTE: intentionally no manifest.json

    cp = subprocess.run(
        [
            sys.executable,
            "-m",
            "diagram2code",
            "benchmark",
            "--dataset",
            str(ds_root),
            "--predictor",
            "oracle",
            "--fail-on-missing-manifest",
        ],
        capture_output=True,
        text=True,
    )
    assert cp.returncode != 0
    assert "no manifest.json" in (cp.stdout + cp.stderr).lower()


def test_cli_benchmark_fail_on_missing_manifest_allows_when_present(tmp_path: Path) -> None:
    ds_root = tmp_path / "ds"
    ds_root.mkdir()
    _make_minimal_dataset_json(ds_root)
    _make_minimal_manifest_json(ds_root)

    cp = subprocess.run(
        [
            sys.executable,
            "-m",
            "diagram2code",
            "benchmark",
            "--dataset",
            str(ds_root),
            "--predictor",
            "oracle",
            "--fail-on-missing-manifest",
        ],
        capture_output=True,
        text=True,
    )

    # This should now proceed into the runner. If oracle needs more dataset structure
    # and fails for other reasons, it will return non-zero. So we assert it does NOT
    # fail with the strict-manifest error message.
    combined = (cp.stdout + cp.stderr).lower()
    assert "no manifest.json" not in combined
