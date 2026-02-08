from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ArtifactType = Literal["zip", "tar.gz", "file", "hf_snapshot"]


@dataclass(frozen=True, slots=True)
class Artifact:
    """
    A downloadable unit used to materialize a dataset version.

    sha256 is mandatory to guarantee reproducibility.
    """

    url: str
    sha256: str | None = None
    type: ArtifactType = "file"
    size_bytes: int | None = None
    strip_top_level: bool = True
    target_subdir: str = "raw"

    # Extraction behavior (only relevant for archives)
    strip_top_level: bool = True

    # Where to place this artifact under the dataset version dir.
    # Examples: "raw", "prepared"
    target_subdir: str = "raw"


@dataclass(frozen=True, slots=True)
class DatasetDescriptor:
    """
    Immutable definition of a remote dataset version.

    This is the reproducibility contract. If "latest" changes, that should be a
    new diagram2code release updating the descriptor.
    """

    name: str
    version: str
    description: str

    homepage: str | None = None
    license: str | None = None

    artifacts: tuple[Artifact, ...] = ()

    # Post-fetch expectations (kept lightweight; strict validation later).
    # Example: ("graphs", "images") or ("dataset.json",)
    expected_layout: tuple[str, ...] = ()

    # Optional hook for later adapters/normalizers without coupling fetcher core.
    loader_hint: str | None = None

    def id(self) -> str:
        """Stable identifier for cache paths and manifests."""
        return f"{self.name}@{self.version}"
