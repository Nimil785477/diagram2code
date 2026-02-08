from __future__ import annotations

from pathlib import Path

import pytest

from diagram2code.datasets.fetching.errors import ManifestError
from diagram2code.datasets.fetching.manifest import (
    MANIFEST_SCHEMA_VERSION,
    DatasetManifestV1,
    ManifestArtifact,
    read_manifest,
    write_manifest,
)


def test_manifest_roundtrip(tmp_path: Path) -> None:
    ds_dir = tmp_path / "flowlearn" / "1.0.0"
    ds_dir.mkdir(parents=True)

    m = DatasetManifestV1(
        schema_version=MANIFEST_SCHEMA_VERSION,
        name="flowlearn",
        version="1.0.0",
        fetched_at_utc="2026-02-08T00:00:00Z",
        artifacts=(
            ManifestArtifact(
                url="https://example.com/a.zip",
                sha256="abc",
                bytes=123,
                local_path="raw/a.zip",
            ),
        ),
        tooling={"diagram2code_version": "0.1.5", "python": "3.x"},
    )

    write_manifest(ds_dir, m)
    got = read_manifest(ds_dir)

    assert got == m


def test_manifest_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ManifestError):
        _ = read_manifest(tmp_path)


def test_manifest_schema_version_mismatch(tmp_path: Path) -> None:
    ds_dir = tmp_path / "x"
    ds_dir.mkdir()

    (ds_dir / "manifest.json").write_text(
        """{
  "schema_version": 999,
  "name": "x",
  "version": "1",
  "fetched_at_utc": "Z",
  "artifacts": [],
  "tooling": {}
}""",
        encoding="utf-8",
    )

    with pytest.raises(ManifestError):
        _ = read_manifest(ds_dir)
