from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
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
from .util import format_bytes


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
    writes manifest.json, performs optional post-processing (build dataset.json/splits.json),
    and returns the dataset dir.
    """
    datasets_root = (cache_root / "datasets") if cache_root is not None else get_cache_root()
    ds_dir = (datasets_root / descriptor.name / descriptor.version).resolve()

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

            # Download snapshot
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

    # Post-process (write dataset.json/splits.json) based on descriptor.loader_hint
    _postprocess_dataset(descriptor, ds_dir)

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

    Includes a tqdm compatibility shim: some hub versions pass `name=...`
    to tqdm, but older tqdm versions don't support it.
    """
    try:
        from huggingface_hub import HfApi, snapshot_download
    except Exception as e:  # pragma: no cover
        raise ArtifactDownloadError(
            "huggingface_hub is required for hf_snapshot artifacts. "
            "Install with: pip install 'diagram2code[hf]'"
        ) from e

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Best-effort: repo metadata
    num_files: int | None = None
    total_bytes: int | None = None
    try:
        api = HfApi()
        info = api.repo_info(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            files_metadata=True,
        )
        siblings = getattr(info, "siblings", None) or []
        num_files = len(siblings)
        sizes = [getattr(s, "size", None) for s in siblings]
        size_ints = [s for s in sizes if isinstance(s, int)]
        total_bytes = sum(size_ints) if size_ints else None
    except Exception:
        num_files = None
        total_bytes = None

    if num_files is None:
        print(f"Fetching HF snapshot: {repo_id}@{revision}")
    else:
        print(
            "Fetching HF snapshot: "
            f"{repo_id}@{revision}\n"
            f"Files: {num_files} | Total: {format_bytes(total_bytes)}"
        )

    # tqdm compat wrapper: drop unsupported kwargs like `name=...`
    tqdm_class = None
    try:
        from tqdm.auto import tqdm as tqdm_auto

        class _TqdmCompat(tqdm_auto):  # type: ignore[misc]
            def __init__(self, *args, **kwargs):
                kwargs.pop("name", None)  # huggingface_hub may pass this
                super().__init__(*args, **kwargs)

        tqdm_class = _TqdmCompat
    except Exception:
        tqdm_class = None

    # Some hub versions may not accept tqdm_class; be backward compatible.
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=str(dest_dir),
            tqdm_class=tqdm_class,
        )
    except TypeError:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=str(dest_dir),
        )


def _postprocess_dataset(descriptor: DatasetDescriptor, ds_dir: Path) -> None:
    """
    Optional post-processing to make raw remote datasets usable by diagram2code.

    For FlowLearn HF snapshot:
      - write dataset.json
      - write splits.json
    """
    hint = (descriptor.loader_hint or "").strip()

    if hint == "flowlearn_hf_snapshot_v1":
        _build_flowlearn_hf_snapshot_layout(ds_dir)
        return

    # tiny_remote_v1 is only for manifest smoke tests; no postprocess needed.
    return


def _build_flowlearn_hf_snapshot_layout(ds_dir: Path) -> None:
    """
    Build a diagram2code-compatible dataset layout for FlowLearn HF snapshot.

    Expected HF layout (confirmed by your listing):
      {ds_dir}/raw/FlowLearn/SciFlowchart/images/*.png
      {ds_dir}/raw/FlowLearn/SciFlowchart/{train,test,all}.json
      {ds_dir}/raw/FlowLearn/SimFlowchart/images/*.png
      {ds_dir}/raw/FlowLearn/SimFlowchart/{train,test,all}.json

    Writes:
      {ds_dir}/dataset.json
      {ds_dir}/splits.json
    """
    raw_root = ds_dir / "raw" / "FlowLearn"
    if not raw_root.exists():
        raise ArtifactDownloadError(f"FlowLearn snapshot root not found: {raw_root}")

    # If already built, keep deterministic (but allow rebuild if files missing)
    dataset_json_path = ds_dir / "dataset.json"
    splits_json_path = ds_dir / "splits.json"

    subsets = ["SciFlowchart", "SimFlowchart"]

    # Collect all images and map stems -> relpath
    stem_to_rel: dict[str, str] = {}
    all_samples: list[dict[str, Any]] = []

    for subset in subsets:
        img_dir = raw_root / subset / "images"
        if not img_dir.exists():
            # Some subsets might be missing on some revisions; skip gracefully
            continue

        for p in sorted(img_dir.glob("*.png")):
            # Ignore weird cache metadata etc. (real images have .png and non-trivial size)
            if p.name.endswith(".metadata"):
                continue
            rel = p.relative_to(ds_dir).as_posix()
            stem_to_rel[p.stem] = rel
            all_samples.append(
                {
                    "id": f"{subset}:{p.stem}",
                    "image_path": rel,
                    "subset": subset,
                }
            )

    if not all_samples:
        raise ArtifactDownloadError(
            f"No FlowLearn images found under: {raw_root} (expected */images/*.png)"
        )

    # Build splits using train/test/all.json if available
    def _read_split_ids(subset: str, split_name: str) -> set[str]:
        fp = raw_root / subset / f"{split_name}.json"
        if not fp.exists():
            return set()

        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            return set()

        # The JSON structure can vary; attempt to extract png-ish identifiers.
        ids: set[str] = set()

        def _visit(x: Any) -> None:
            if isinstance(x, dict):
                for v in x.values():
                    _visit(v)
            elif isinstance(x, list):
                for v in x:
                    _visit(v)
            elif isinstance(x, str):
                # Candidate: png filename or stem.
                if x.lower().endswith(".png"):
                    ids.add(Path(x).stem)
                else:
                    # Some records may store ids without extension (common in datasets)
                    # Only accept if it looks like our stems (cheap filter)
                    if x in stem_to_rel:
                        ids.add(x)

        _visit(data)
        return ids

    train_stems: set[str] = set()
    test_stems: set[str] = set()

    for subset in subsets:
        train_stems |= _read_split_ids(subset, "train")
        test_stems |= _read_split_ids(subset, "test")

    # If split jsons didn’t yield anything, fallback to "all goes to test" (safe default)
    if not train_stems and not test_stems:
        # Put everything in "test" so benchmarking works immediately,
        # without implying a meaningful train split.
        test_ids = [s["id"] for s in all_samples]
        splits = {"test": test_ids}
    else:
        train_ids: list[str] = []
        test_ids: list[str] = []

        for s in all_samples:
            stem = s["id"].split(":", 1)[1]
            if stem in train_stems:
                train_ids.append(s["id"])
            if stem in test_stems:
                test_ids.append(s["id"])

        # Ensure non-empty fallback
        if not test_ids:
            test_ids = [s["id"] for s in all_samples if s["id"] not in train_ids]
        if not train_ids:
            # Keep train optional; tests usually care that split exists and loads
            pass

        splits = {}
        if train_ids:
            splits["train"] = sorted(train_ids)
        if test_ids:
            splits["test"] = sorted(test_ids)

    dataset_json = {
        "schema_version": 1,
        "name": "flowlearn",
        "description": "FlowLearn HF snapshot converted to diagram2code dataset format.",
        "root": ".",
        "samples": all_samples,
    }
    splits_json = {
        "schema_version": 1,
        "splits": splits,
    }

    dataset_json_path.write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")
    splits_json_path.write_text(json.dumps(splits_json, indent=2), encoding="utf-8")
