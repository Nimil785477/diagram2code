from __future__ import annotations

from diagram2code.cli import main


def test_dataset_clean_single_version(monkeypatch, tmp_path):
    monkeypatch.setenv("DIAGRAM2CODE_CACHE_DIR", str(tmp_path))

    ds = tmp_path / "datasets" / "tiny_remote_v1" / "1"
    ds.mkdir(parents=True)

    rc = main(
        [
            "dataset",
            "clean",
            "tiny_remote_v1",
            "--yes",
        ]
    )

    assert rc == 0
    assert not ds.exists()


def test_dataset_clean_all_versions(monkeypatch, tmp_path):
    monkeypatch.setenv("DIAGRAM2CODE_CACHE_DIR", str(tmp_path))

    root = tmp_path / "datasets" / "tiny_remote_v1"
    (root / "1").mkdir(parents=True)
    (root / "2").mkdir(parents=True)

    rc = main(
        [
            "dataset",
            "clean",
            "tiny_remote_v1",
            "--all",
            "--yes",
        ]
    )

    assert rc == 0
    assert not root.exists()
