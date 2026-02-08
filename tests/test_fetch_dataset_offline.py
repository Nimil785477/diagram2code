from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from diagram2code.datasets.fetching.descriptors import Artifact, DatasetDescriptor
from diagram2code.datasets.fetching.errors import ArtifactDownloadError, HashMismatchError
from diagram2code.datasets.fetching.fetcher import fetch_dataset
from diagram2code.datasets.fetching.manifest import read_manifest


class FakeDownloader:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def download_to_path(self, url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.payload)


class ExplodingDownloader:
    def download_to_path(self, url: str, dest: Path) -> None:
        raise RuntimeError("boom")


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_fetch_dataset_writes_manifest_and_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"hello"
    sha = _sha256(payload)

    desc = DatasetDescriptor(
        name="flowlearn",
        version="1.0.0",
        description="x",
        artifacts=(
            Artifact(
                url="https://example.com/a.bin",
                sha256=sha,
                type="file",
                target_subdir="raw",
            ),
        ),
    )

    ds_dir = fetch_dataset(
        desc,
        cache_root=tmp_path / "cache",
        downloader=FakeDownloader(payload),
    )

    assert (ds_dir / "raw" / "a.bin").read_bytes() == payload

    m = read_manifest(ds_dir)
    assert m.name == "flowlearn"
    assert m.version == "1.0.0"
    assert len(m.artifacts) == 1
    assert m.artifacts[0].sha256 == sha
    assert m.artifacts[0].local_path == "raw/a.bin"


def test_fetch_dataset_hash_mismatch_raises(tmp_path: Path) -> None:
    payload = b"hello"
    desc = DatasetDescriptor(
        name="x",
        version="1",
        description="x",
        artifacts=(Artifact(url="https://example.com/a.bin", sha256="deadbeef", type="file"),),
    )

    with pytest.raises(HashMismatchError):
        _ = fetch_dataset(
            desc,
            cache_root=tmp_path / "cache",
            downloader=FakeDownloader(payload),
        )


def test_fetch_dataset_download_error_raises(tmp_path: Path) -> None:
    payload = b"hello"
    sha = _sha256(payload)

    desc = DatasetDescriptor(
        name="x",
        version="1",
        description="x",
        artifacts=(Artifact(url="https://example.com/a.bin", sha256=sha, type="file"),),
    )

    with pytest.raises(ArtifactDownloadError):
        _ = fetch_dataset(
            desc,
            cache_root=tmp_path / "cache",
            downloader=ExplodingDownloader(),
        )
