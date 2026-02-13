from __future__ import annotations

import hashlib
import json
from pathlib import Path

from diagram2code.datasets.fetching.cache import dataset_dir
from diagram2code.datasets.fetching.errors import (
    DatasetFetchError,
    DatasetNotFoundError,
    ManifestError,
)
from diagram2code.datasets.fetching.fetcher import fetch_dataset
from diagram2code.datasets.fetching.manifest import read_manifest
from diagram2code.datasets.fetching.registry import RemoteDatasetRegistry


def dataset_list_cmd() -> int:
    reg = RemoteDatasetRegistry.builtins()
    for name in reg.list():
        print(name)
    return 0


def dataset_fetch_cmd(
    name: str,
    *,
    force: bool,
    cache_dir: Path | None,
    yes: bool,
) -> int:
    reg = RemoteDatasetRegistry.builtins()
    try:
        desc = reg.get(name)
    except DatasetNotFoundError as e:
        print(str(e))
        return 2

    # Safety: prevent accidental huge downloads unless explicitly confirmed
    if any(a.type == "hf_snapshot" for a in desc.artifacts) and not yes:
        print(
            "Refusing to fetch without confirmation: this dataset may be large.\n"
            f"Re-run with: diagram2code dataset fetch {name} --yes"
        )
        return 2

    try:
        ds_dir = fetch_dataset(desc, cache_root=cache_dir, force=force)
    except DatasetFetchError as e:
        print(str(e))
        return 2

    print(str(ds_dir))
    return 0


def dataset_path_cmd(name: str, *, cache_dir: Path | None) -> int:
    reg = RemoteDatasetRegistry.builtins()
    try:
        desc = reg.get(name)
    except DatasetNotFoundError as e:
        print(str(e))
        return 2

    p = (
        (cache_dir / "datasets" / desc.name / desc.version)
        if cache_dir
        else dataset_dir(desc.name, desc.version)
    )

    if not p.exists():
        print(f"Dataset not installed: {name}")
        return 2

    print(str(p))
    return 0


def dataset_info_cmd(name: str, *, cache_dir: Path | None, installed_only: bool = False) -> int:
    reg = RemoteDatasetRegistry.builtins()
    try:
        desc = reg.get(name)
    except DatasetNotFoundError as e:
        print(str(e))
        return 2

    p = (
        (cache_dir / "datasets" / desc.name / desc.version)
        if cache_dir
        else dataset_dir(desc.name, desc.version)
    )

    installed = p.exists()
    manifest_path = p / "manifest.json"
    manifest_sha256 = None
    manifest_summary = None

    if installed and manifest_path.exists():
        raw = manifest_path.read_bytes()
        manifest_sha256 = hashlib.sha256(raw).hexdigest()

        # Parse manifest and summarize it
        m = read_manifest(p)
        manifest_summary = {
            "schema_version": m.schema_version,
            "name": m.name,
            "version": m.version,
            "fetched_at_utc": m.fetched_at_utc,
            "tooling": m.tooling,
            "artifacts": [
                {
                    "url": a.url,
                    "sha256": a.sha256,
                    "bytes": a.bytes,
                    "local_path": a.local_path,
                }
                for a in m.artifacts
            ],
        }
    if installed_only and not installed:
        print(f"Dataset not installed: {name}")
        return 2

    info = {
        "name": desc.name,
        "version": desc.version,
        "description": desc.description,
        "homepage": desc.homepage,
        "installed": installed,
        "manifest_path": str(manifest_path) if installed and manifest_path.exists() else None,
        "manifest_sha256": manifest_sha256,
        "manifest": manifest_summary,
        "path": str(p) if installed else None,
        "artifacts": [
            {
                "url": a.url,
                "type": a.type,
                "sha256": a.sha256,
                "target_subdir": a.target_subdir,
            }
            for a in desc.artifacts
        ],
    }

    print(json.dumps(info, indent=2, sort_keys=True))
    return 0


def dataset_verify_cmd(name: str, *, cache_dir: Path | None, deep: bool = False) -> int:
    reg = RemoteDatasetRegistry.builtins()
    try:
        desc = reg.get(name)
    except DatasetNotFoundError as e:
        print(str(e))
        return 2

    p = (
        (cache_dir / "datasets" / desc.name / desc.version)
        if cache_dir
        else dataset_dir(desc.name, desc.version)
    )

    try:
        m = read_manifest(p)
    except (ManifestError, OSError) as e:
        print(str(e))
        return 2

    # Identity check (existing behavior)
    ok = (m.name == desc.name) and (m.version == desc.version)
    if not ok:
        print(
            "Manifest does not match descriptor.\n"
            f"manifest: name={m.name!r} version={m.version!r}\n"
            f"expected: name={desc.name!r} version={desc.version!r}"
        )
        return 2
    if not p.exists():
        print(f"Dataset not installed: {name}")
        return 2

    # Ensure manifest artifacts match descriptor artifacts (URL + sha256/revision)
    manifest_by_url = {a.url: a for a in m.artifacts}
    for a in desc.artifacts:
        ma = manifest_by_url.get(a.url)
        if ma is None:
            print(f"Missing artifact in manifest: {a.url}")
            return 2
        if a.sha256 is not None and ma.sha256 != a.sha256:
            print(
                "Artifact sha256 mismatch.\n"
                f"url: {a.url}\n"
                f"manifest: {ma.sha256}\n"
                f"expected: {a.sha256}"
            )
            return 2

    if not deep:
        print("OK")
        return 0

    # Deep verification:
    # - file artifacts: re-hash and compare
    # - hf_snapshot: ensure directory exists and is non-empty
    for a in desc.artifacts:
        ma = manifest_by_url[a.url]
        local_path = p / Path(ma.local_path)

        if a.type == "hf_snapshot":
            if not local_path.exists() or not local_path.is_dir():
                print(f"Missing snapshot directory: {local_path}")
                return 2
            try:
                has_any = any(local_path.rglob("*"))
            except OSError:
                has_any = False
            if not has_any:
                print(f"Snapshot directory is empty: {local_path}")
                return 2
            continue

        # Default: treat as file artifact
        if not local_path.exists() or not local_path.is_file():
            print(f"Missing artifact file: {local_path}")
            return 2

        expected = a.sha256
        if expected is None:
            print(f"Missing expected sha256 for artifact: {a.url}")
            return 2

        got = _sha256_file(local_path)
        if got != expected:
            print(f"sha256 mismatch.\npath: {local_path}\nexpected: {expected}\ngot: {got}")
            return 2

    print("OK")
    return 0


def dataset_clean_cmd(
    name: str,
    *,
    all_versions: bool,
    yes: bool,
    cache_dir: Path | None,
) -> int:
    from diagram2code.datasets.fetching.cache import get_cache_root
    from diagram2code.datasets.fetching.errors import DatasetNotFoundError
    from diagram2code.datasets.fetching.registry import RemoteDatasetRegistry

    reg = RemoteDatasetRegistry.builtins()
    try:
        desc = reg.get(name)
    except DatasetNotFoundError as e:
        print(str(e))
        return 2

    datasets_root = (cache_dir / "datasets") if cache_dir else get_cache_root()

    targets: list[Path] = []

    if all_versions:
        root = datasets_root / desc.name
        if root.exists():
            targets = [p for p in root.iterdir() if p.is_dir()]
    else:
        p = datasets_root / desc.name / desc.version
        if p.exists():
            targets = [p]

    if not targets:
        print(f"No installed dataset found for: {name}")
        return 0

    print("The following dataset directories will be removed:")
    for t in targets:
        print(f"  - {t}")

    if not yes:
        resp = input("Proceed? [y/N]: ").strip().lower()
        if resp not in {"y", "yes"}:
            print("Aborted.")
            return 1

    for t in targets:
        _rm_tree(t)

    if all_versions:
        dataset_root = datasets_root / desc.name
        try:
            if dataset_root.exists() and not any(dataset_root.iterdir()):
                dataset_root.rmdir()
        except OSError:
            pass

    print(f"Removed {len(targets)} dataset directory(s).")
    return 0


def _rm_tree(path: Path) -> None:
    # Windows-friendly recursive delete
    if not path.exists():
        return
    for p in sorted(path.rglob("*"), reverse=True):
        if p.is_file() or p.is_symlink():
            p.unlink()
        elif p.is_dir():
            p.rmdir()
    path.rmdir()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
