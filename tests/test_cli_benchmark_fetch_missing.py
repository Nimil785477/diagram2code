from __future__ import annotations

import json

from diagram2code.cli import main


def test_benchmark_missing_registry_dataset_errors(monkeypatch, tmp_path, capsys):
    # Ensure cache is isolated for test
    monkeypatch.setenv("DIAGRAM2CODE_CACHE_DIR", str(tmp_path))

    rc = main(
        [
            "benchmark",
            "--dataset",
            "tiny_remote_v1",
            "--predictor",
            "oracle",
        ]
    )
    out = capsys.readouterr().out

    assert rc == 2
    assert "Dataset not installed: tiny_remote_v1" in out
    assert "diagram2code dataset fetch tiny_remote_v1" in out
    assert "--fetch-missing" in out


def test_benchmark_fetch_missing_registry_dataset(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("DIAGRAM2CODE_CACHE_DIR", str(tmp_path))

    # Import registry/paths so we know where the fetch should land.
    from diagram2code.datasets.fetching.cache import dataset_dir
    from diagram2code.datasets.fetching.registry import RemoteDatasetRegistry

    desc = RemoteDatasetRegistry.builtins().get("tiny_remote_v1")
    target_dir = dataset_dir(desc.name, desc.version)
    manifest_path = target_dir / "manifest.json"

    # Stub fetch_dataset to avoid network and simulate a successful install.
    def _fake_fetch_dataset(desc, cache_root=None, force=False):
        target_dir.mkdir(parents=True, exist_ok=True)

        # Use the known-good example dataset schema as the minimal valid layout.
        from diagram2code.datasets import DatasetRegistry

        example_root = DatasetRegistry().resolve_root("example:minimal_v1")

        # Copy required dataset layout files from example:minimal_v1
        # (This avoids guessing schema details.)
        for fname in ["dataset.json", "splits.json"]:
            src = example_root / fname
            if src.exists():
                (target_dir / fname).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

        # Copy referenced files/dirs used by the example dataset.
        # We conservatively copy everything except the core JSON files.
        import shutil

        for child in example_root.iterdir():
            if child.name in {"dataset.json", "splits.json"}:
                continue
            dst = target_dir / child.name
            if child.is_dir():
                shutil.copytree(child, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(child, dst)

        # Also write manifest.json so Phase 6 verification expectations remain realistic
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "name": desc.name,
                    "version": desc.version,
                    "fetched_at_utc": "2026-02-10T00:00:00Z",
                    "artifacts": [],
                    "tooling": {},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return target_dir

    monkeypatch.setattr(
        "diagram2code.datasets.fetching.fetcher.fetch_dataset",
        _fake_fetch_dataset,
    )

    out_json = tmp_path / "result.json"

    rc = main(
        [
            "benchmark",
            "--dataset",
            "tiny_remote_v1",
            "--predictor",
            "oracle",
            "--fetch-missing",
            "--yes",
            "--json",
            str(out_json),
        ]
    )

    assert rc == 0
    assert target_dir.exists()
    assert manifest_path.exists()
    assert out_json.exists()
