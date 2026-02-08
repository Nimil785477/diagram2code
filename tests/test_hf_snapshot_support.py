from __future__ import annotations

from pathlib import Path

import pytest

from diagram2code.datasets.fetching.descriptors import Artifact, DatasetDescriptor
from diagram2code.datasets.fetching.errors import ArtifactDownloadError
from diagram2code.datasets.fetching.fetcher import fetch_dataset


def test_hf_snapshot_requires_sha256_revision(tmp_path: Path) -> None:
    desc = DatasetDescriptor(
        name="x",
        version="1",
        description="x",
        artifacts=(
            Artifact(
                url="hf://datasets/jopan/FlowLearn@abc",
                sha256=None,
                type="hf_snapshot",
                target_subdir="raw",
            ),
        ),
    )

    with pytest.raises(ArtifactDownloadError, match="must set sha256"):
        _ = fetch_dataset(desc, cache_root=tmp_path / "cache")


def test_hf_snapshot_revision_must_match_sha256(tmp_path: Path) -> None:
    desc = DatasetDescriptor(
        name="x",
        version="1",
        description="x",
        artifacts=(
            Artifact(
                url="hf://datasets/jopan/FlowLearn@revA",
                sha256="revB",
                type="hf_snapshot",
                target_subdir="raw",
            ),
        ),
    )

    with pytest.raises(ArtifactDownloadError, match="revision mismatch"):
        _ = fetch_dataset(desc, cache_root=tmp_path / "cache")


def test_hf_snapshot_calls_downloader_hook(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Monkeypatch the internal hf download function so no network / dependency is needed.
    import diagram2code.datasets.fetching.fetcher as fetcher_mod

    def fake_download(repo_id: str, revision: str, dest_dir: Path) -> None:
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / "README.md").write_text("ok", encoding="utf-8")

    monkeypatch.setattr(fetcher_mod, "_hf_snapshot_download", fake_download)

    rev = "35d7dc891fd0ac17f3773aeeba023fe15acbd062"
    desc = DatasetDescriptor(
        name="flowlearn",
        version="hf-test",
        description="x",
        artifacts=(
            Artifact(
                url=f"hf://datasets/jopan/FlowLearn@{rev}",
                sha256=rev,
                type="hf_snapshot",
                target_subdir="raw",
            ),
        ),
    )

    ds_dir = fetch_dataset(desc, cache_root=tmp_path / "cache")
    assert (ds_dir / "raw" / "FlowLearn" / "README.md").read_text(encoding="utf-8") == "ok"
