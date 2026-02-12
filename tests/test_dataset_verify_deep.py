from __future__ import annotations

import hashlib
import json

from diagram2code.cli import main
from diagram2code.datasets.fetching.descriptors import Artifact, DatasetDescriptor


def _sha(s: bytes) -> str:
    return hashlib.sha256(s).hexdigest()


def test_dataset_verify_deep_ok(monkeypatch, tmp_path):
    # Fake registry with a deterministic artifact
    desc = DatasetDescriptor(
        name="testds",
        version="1",
        description="test",
        homepage=None,
        artifacts=(
            Artifact(
                type="file",
                url="https://example.com/data.bin",
                sha256=_sha(b"hello"),
                target_subdir="",
            ),
        ),
    )

    class _FakeReg:
        def get(self, name: str):
            assert name == "testds"
            return desc

        def list(self):
            return ["testds"]

    from diagram2code.datasets.fetching import registry as reg_mod

    monkeypatch.setattr(
        reg_mod.RemoteDatasetRegistry, "builtins", classmethod(lambda cls: _FakeReg())
    )

    # Install into cache-dir layout
    ds_root = tmp_path / "datasets" / "testds" / "1"
    ds_root.mkdir(parents=True)

    # Write the artifact file at the manifest path
    (ds_root / "data.bin").write_bytes(b"hello")

    # Write manifest matching descriptor + local_path
    (ds_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "testds",
                "version": "1",
                "fetched_at_utc": "2026-02-10T00:00:00Z",
                "artifacts": [
                    {
                        "url": "https://example.com/data.bin",
                        "sha256": _sha(b"hello"),
                        "bytes": 5,
                        "local_path": "data.bin",
                    }
                ],
                "tooling": {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rc = main(["dataset", "verify", "testds", "--deep", "--cache-dir", str(tmp_path)])
    assert rc == 0


def test_dataset_verify_deep_hash_mismatch(monkeypatch, tmp_path):
    desc = DatasetDescriptor(
        name="testds",
        version="1",
        description="test",
        homepage=None,
        artifacts=(
            Artifact(
                type="file",
                url="https://example.com/data.bin",
                sha256=_sha(b"hello"),
                target_subdir="",
            ),
        ),
    )

    class _FakeReg:
        def get(self, name: str):
            return desc

        def list(self):
            return ["testds"]

    from diagram2code.datasets.fetching import registry as reg_mod

    monkeypatch.setattr(
        reg_mod.RemoteDatasetRegistry, "builtins", classmethod(lambda cls: _FakeReg())
    )

    ds_root = tmp_path / "datasets" / "testds" / "1"
    ds_root.mkdir(parents=True)

    # Wrong contents
    (ds_root / "data.bin").write_bytes(b"oops")

    (ds_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1.1,
                "name": "testds",
                "version": "1",
                "fetched_at_utc": "2026-02-10T00:00:00Z",
                "artifacts": [
                    {
                        "url": "https://example.com/data.bin",
                        "sha256": _sha(b"hello"),
                        "bytes": 4,
                        "local_path": "data.bin",
                    }
                ],
                "tooling": {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rc = main(["dataset", "verify", "testds", "--deep", "--cache-dir", str(tmp_path)])
    assert rc == 2
