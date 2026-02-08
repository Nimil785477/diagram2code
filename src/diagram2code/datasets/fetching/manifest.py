from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .errors import ManifestError

MANIFEST_FILENAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ManifestArtifact:
    url: str
    sha256: str
    bytes: int | None
    local_path: str


@dataclass(frozen=True, slots=True)
class DatasetManifestV1:
    schema_version: int
    name: str
    version: str
    fetched_at_utc: str
    artifacts: tuple[ManifestArtifact, ...]
    tooling: dict[str, str]


def manifest_path(dataset_dir: Path) -> Path:
    return dataset_dir / MANIFEST_FILENAME


def write_manifest(dataset_dir: Path, manifest: DatasetManifestV1) -> None:
    path = manifest_path(dataset_dir)
    try:
        path.write_text(
            json.dumps(asdict(manifest), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except OSError as e:
        raise ManifestError(f"Failed to write manifest: {path}") from e


def read_manifest(dataset_dir: Path) -> DatasetManifestV1:
    path = manifest_path(dataset_dir)

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ManifestError(f"Manifest not found or unreadable: {path}") from e

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ManifestError(f"Manifest is not valid JSON: {path}") from e

    _validate_manifest_dict(data, path)

    artifacts = tuple(
        ManifestArtifact(
            url=a["url"],
            sha256=a["sha256"],
            bytes=a.get("bytes"),
            local_path=a["local_path"],
        )
        for a in data["artifacts"]
    )

    return DatasetManifestV1(
        schema_version=data["schema_version"],
        name=data["name"],
        version=data["version"],
        fetched_at_utc=data["fetched_at_utc"],
        artifacts=artifacts,
        tooling=dict(data["tooling"]),
    )


def _validate_manifest_dict(data: Any, path: Path) -> None:
    if not isinstance(data, dict):
        raise ManifestError(f"Manifest must be a JSON object: {path}")

    if data.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            "Unsupported manifest schema_version: "
            f"{data.get('schema_version')} "
            f"(expected {MANIFEST_SCHEMA_VERSION})"
        )

    for key in ("name", "version", "fetched_at_utc", "artifacts", "tooling"):
        if key not in data:
            raise ManifestError(f"Manifest missing key '{key}': {path}")

    if not isinstance(data["artifacts"], list):
        raise ManifestError(f"Manifest 'artifacts' must be a list: {path}")

    for i, artifact in enumerate(data["artifacts"]):
        if not isinstance(artifact, dict):
            raise ManifestError(f"Manifest artifacts[{i}] must be an object: {path}")
        for k in ("url", "sha256", "local_path"):
            if k not in artifact:
                raise ManifestError(f"Manifest artifacts[{i}] missing '{k}': {path}")

    if not isinstance(data["tooling"], dict):
        raise ManifestError(f"Manifest 'tooling' must be an object: {path}")
