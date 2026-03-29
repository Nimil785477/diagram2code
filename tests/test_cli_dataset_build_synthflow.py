from __future__ import annotations

from pathlib import Path

from diagram2code.cli import main


def test_cli_dataset_build_synthflow(tmp_path: Path) -> None:
    out = tmp_path / "synthflow_cli"

    rc = main(
        [
            "dataset",
            "build",
            "synthflow",
            "--out",
            str(out),
            "--split",
            "test",
            "--num-samples",
            "3",
            "--seed",
            "0",
        ]
    )

    assert rc == 0
    assert (out / "dataset.json").exists()
    assert (out / "splits.json").exists()
    assert (out / "images").is_dir()
    assert (out / "graphs").is_dir()
    assert len(list((out / "images").glob("*.png"))) == 3
    assert len(list((out / "graphs").glob("*.json"))) == 3
