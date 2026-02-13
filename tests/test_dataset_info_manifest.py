from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from diagram2code.datasets.fetching.manifest import MANIFEST_SCHEMA_VERSION
from diagram2code.datasets.fetching.registry import RemoteDatasetRegistry


def test_dataset_info_includes_manifest_summary_when_installed(tmp_path: Path) -> None:
    reg = RemoteDatasetRegistry.builtins()
    names = sorted(reg.list())
    assert names, "RemoteDatasetRegistry.builtins() returned no datasets"
    name = names[0]
    desc = reg.get(name)

    cache_dir = tmp_path / "cache"
    ds_root = cache_dir / "datasets" / desc.name / desc.version
    ds_root.mkdir(parents=True, exist_ok=True)

    # Create a dummy artifact file referenced by the manifest
    artifact_rel = Path("raw") / "a.zip"
    artifact_path = ds_root / artifact_rel
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_bytes = b"hello-diagram2code"
    artifact_path.write_bytes(artifact_bytes)
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()

    manifest_obj = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "name": desc.name,
        "version": desc.version,
        "fetched_at_utc": "2026-02-08T00:00:00Z",
        "artifacts": [
            {
                "url": "https://example.com/a.zip",
                "sha256": artifact_sha256,
                "bytes": len(artifact_bytes),
                "local_path": str(artifact_rel).replace("\\", "/"),
            }
        ],
        "tooling": {"diagram2code_version": "0.1.6", "python": "3.x"},
    }

    manifest_path = ds_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_obj, indent=2), encoding="utf-8")

    expected_manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    cp = subprocess.run(
        [
            sys.executable,
            "-m",
            "diagram2code",
            "dataset",
            "info",
            name,
            "--cache-dir",
            str(cache_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert cp.returncode == 0, cp.stderr

    data = json.loads(cp.stdout)
    assert data["name"] == desc.name
    assert data["version"] == desc.version
    assert data["installed"] is True
    assert data["path"] is not None

    assert data["manifest_path"] is not None
    assert data["manifest_sha256"] == expected_manifest_sha256

    m = data["manifest"]
    assert isinstance(m, dict)
    assert m["name"] == desc.name
    assert m["version"] == desc.version
    assert "artifacts" in m
    assert isinstance(m["artifacts"], list)
    assert len(m["artifacts"]) == 1
    assert m["artifacts"][0]["local_path"].endswith("raw/a.zip")


def test_dataset_info_no_manifest_fields_when_not_installed(tmp_path: Path) -> None:
    reg = RemoteDatasetRegistry.builtins()
    names = sorted(reg.list())
    assert names, "RemoteDatasetRegistry.builtins() returned no datasets"
    name = names[0]
    desc = reg.get(name)

    cache_dir = tmp_path / "cache"  # intentionally empty

    cp = subprocess.run(
        [
            sys.executable,
            "-m",
            "diagram2code",
            "dataset",
            "info",
            name,
            "--cache-dir",
            str(cache_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert cp.returncode == 0, cp.stderr

    data = json.loads(cp.stdout)
    assert data["name"] == desc.name
    assert data["version"] == desc.version
    assert data["installed"] is False
    assert data["path"] is None

    # stable shape: keys exist but are null when not installed
    assert data.get("manifest_path") is None
    assert data.get("manifest_sha256") is None
    assert data.get("manifest") is None
