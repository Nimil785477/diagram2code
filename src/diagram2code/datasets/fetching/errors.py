from __future__ import annotations


class DatasetFetchError(Exception):
    """Base error for dataset fetching / caching / verification."""


class DatasetNotFoundError(DatasetFetchError):
    """Requested remote dataset name/version does not exist in the registry."""


class HashMismatchError(DatasetFetchError):
    """Artifact hash does not match the expected sha256."""


class ArtifactDownloadError(DatasetFetchError):
    """Downloading an artifact failed."""


class ArtifactExtractError(DatasetFetchError):
    """Extracting an archive artifact failed."""


class ManifestError(DatasetFetchError):
    """Manifest is missing, unreadable, or invalid."""
