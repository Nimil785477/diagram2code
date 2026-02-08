from __future__ import annotations

import hashlib
import platform
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from urllib.error import URLError

from diagram2code import __version__ as diagram2code_version

from .cache import get_cache_root
from .descriptors import Artifact, DatasetDescriptor
from .errors import ArtifactDownloadError, HashMismatchError
from .manifest import (
    MANIFEST_SCHEMA_VERSION,
    DatasetManifestV1,
    ManifestArtifact,
    write_manifest,
)


class Downloader(Protocol):
    def download_to_path(self, url: str, dest: Path) -> None:
        """Download the content at url into dest (creating parent dirs if needed)."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class DefaultDownloader:
    """
    Default downloader using stdlib urllib.

    Supports:
      - https://, http://
      - file:///... (useful for offline tests)
    """

    timeout_seconds: float = 60.0

    def download_to_path(self, url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            with urllib.request.urlopen(url, timeout=self.timeout_seconds) as r:
                # Stream to disk (Windows-friendly)
                with dest.open("wb") as f:
                    shutil.copyfileobj(r, f, length=1024 * 1024)
        except (OSError, URLError, ValueError) as e:
            raise ArtifactDownloadError(f"Failed to download: {url}") from e


def fetch_dataset(
    descriptor: DatasetDescriptor,
    cache_root: Path | None = None,
    force: bool = False,
    downloader: Downloader | None = None,
) -> Path:
    """
    Ensure a dataset version exists in the local cache:
      {cache}/{name}/{version}/...

    Downloads artifacts, verifies sha256 (or pinned revision for hf_snapshot),
    writes manifest.json, and returns the dataset dir.
    """
    root = cache_root if cache_root is not None else get_cache_root()
    ds_dir = (root / descriptor.name / descriptor.version).resolve()

    if downloader is None:
        downloader = DefaultDownloader()

    if force and ds_dir.exists():
        _rm_tree(ds_dir)

    ds_dir.mkdir(parents=True, exist_ok=True)

    fetched_at = (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )

    manifest_artifacts: list[ManifestArtifact] = []

    for artifact in descriptor.artifacts:
        if artifact.type == "hf_snapshot":
            if artifact.sha256 is None:
                raise ArtifactDownloadError(
                    "hf_snapshot artifacts must set sha256 to the pinned HF revision (commit hash)."
                )

            try:
                repo_id, revision = _parse_hf_snapshot_url(artifact.url)
            except ValueError as e:
                raise ArtifactDownloadError(str(e)) from e

            # Enforce reproducibility: the URL revision must match artifact.sha256
            if revision != artifact.sha256:
                raise ArtifactDownloadError(
                    "hf_snapshot revision mismatch: "
                    f"url has {revision}, "
                    f"sha256 field has {artifact.sha256}"
                )

            dataset_name = repo_id.split("/", 1)[1]
            target_dir = ds_dir / artifact.target_subdir / dataset_name
            _hf_snapshot_download(repo_id, revision, target_dir)

            rel = Path(artifact.target_subdir) / dataset_name
            manifest_artifacts.append(
                ManifestArtifact(
                    url=artifact.url,
                    sha256=artifact.sha256,
                    bytes=None,
                    local_path=str(rel).replace("\\", "/"),
                )
            )
            continue

        # Non-HF artifacts require a sha256
        if artifact.sha256 is None:
            raise ArtifactDownloadError(
                f"Artifact sha256 is required for type '{artifact.type}': {artifact.url}"
            )

        local_rel, local_abs = _artifact_paths(ds_dir, artifact)

        # Skip re-download if exists and matches hash (unless force)
        if local_abs.exists() and not force:
            if _sha256_file(local_abs) == artifact.sha256:
                manifest_artifacts.append(
                    ManifestArtifact(
                        url=artifact.url,
                        sha256=artifact.sha256,
                        bytes=local_abs.stat().st_size,
                        local_path=str(local_rel).replace("\\", "/"),
                    )
                )
                continue

        local_abs.parent.mkdir(parents=True, exist_ok=True)

        try:
            downloader.download_to_path(artifact.url, local_abs)
        except Exception as e:  # keep protocol flexible
            raise ArtifactDownloadError(f"Failed to download artifact: {artifact.url}") from e

        got = _sha256_file(local_abs)
        if got != artifact.sha256:
            raise HashMismatchError(
                f"sha256 mismatch for {artifact.url}: expected {artifact.sha256}, got {got}"
            )

        manifest_artifacts.append(
            ManifestArtifact(
                url=artifact.url,
                sha256=artifact.sha256,
                bytes=local_abs.stat().st_size,
                local_path=str(local_rel).replace("\\", "/"),
            )
        )

    manifest = DatasetManifestV1(
        schema_version=MANIFEST_SCHEMA_VERSION,
        name=descriptor.name,
        version=descriptor.version,
        fetched_at_utc=fetched_at,
        artifacts=tuple(manifest_artifacts),
        tooling={
            "diagram2code_version": diagram2code_version,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    )

    write_manifest(ds_dir, manifest)
    return ds_dir


def _artifact_paths(ds_dir: Path, artifact: Artifact) -> tuple[Path, Path]:
    """
    Return (relative_path, absolute_path) for the artifact file.
    We name the file using a stable suffix derived from the URL path.
    """
    filename = Path(artifact.url.split("?", 1)[0]).name
    if not filename:
        filename = "artifact.bin"

    rel = Path(artifact.target_subdir) / filename
    return rel, ds_dir / rel


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rm_tree(path: Path) -> None:
    # Local, dependency-free recursive delete (Windows-friendly)
    if not path.exists():
        return
    for p in sorted(path.rglob("*"), reverse=True):
        if p.is_file() or p.is_symlink():
            p.unlink()
        elif p.is_dir():
            p.rmdir()
    path.rmdir()


def _parse_hf_snapshot_url(url: str) -> tuple[str, str]:
    """
    Parse: hf://datasets/<repo_id>@<revision>

    Example:
      hf://datasets/jopan/FlowLearn@35d7dc8...

    Returns: (repo_id, revision)
    """
    prefix = "hf://datasets/"
    if not url.startswith(prefix):
        raise ValueError(f"Invalid HF snapshot url (expected {prefix}...): {url}")

    rest = url[len(prefix) :]
    if "@" not in rest:
        raise ValueError(f"HF snapshot url must include @<revision>: {url}")

    repo_id, revision = rest.split("@", 1)
    repo_id = repo_id.strip()
    revision = revision.strip()

    if not repo_id or "/" not in repo_id:
        raise ValueError(f"Invalid HF repo_id in url: {url}")
    if not revision:
        raise ValueError(f"Invalid HF revision in url: {url}")

    return repo_id, revision


def _hf_snapshot_download(repo_id: str, revision: str, dest_dir: Path) -> None:
    """
    Download a dataset snapshot from Hugging Face Hub into dest_dir.

    Implemented via huggingface_hub.snapshot_download, but imported lazily
    so core installs don't require it unless hf_snapshot is used.
    """
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:  # pragma: no cover
        raise ArtifactDownloadError(
            "huggingface_hub is required for hf_snapshot artifacts. "
            "Install with: pip install 'diagram2code[hf]'"
        ) from e

    dest_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(dest_dir),
    )
