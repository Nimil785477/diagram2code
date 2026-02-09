from __future__ import annotations

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


def dataset_info_cmd(name: str, *, cache_dir: Path | None) -> int:
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

    info = {
        "name": desc.name,
        "version": desc.version,
        "description": desc.description,
        "homepage": desc.homepage,
        "installed": installed,
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


def dataset_verify_cmd(name: str, *, cache_dir: Path | None) -> int:
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

    # Minimal verification: manifest exists + matches dataset identity.
    # Step 8 can upgrade this to full artifact re-hash verification.
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

    print("OK")
    return 0
