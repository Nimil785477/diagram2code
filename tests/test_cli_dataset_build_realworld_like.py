from __future__ import annotations

import json

from diagram2code.cli import main


def test_cli_dataset_build_realworld_like_smoke(tmp_path, capsys) -> None:
    out_dir = tmp_path / "realworld_like_ds"

    rc = main(
        [
            "dataset",
            "build",
            "realworld-like",
            "--out",
            str(out_dir),
            "--split",
            "test",
            "--num-samples",
            "5",
            "--seed",
            "0",
        ]
    )

    captured = capsys.readouterr()

    assert rc == 0
    assert out_dir.exists()
    assert (out_dir / "dataset.json").exists()
    assert (out_dir / "splits.json").exists()
    assert (out_dir / "images").exists()
    assert (out_dir / "graphs").exists()

    payload = json.loads((out_dir / "dataset.json").read_text(encoding="utf-8"))
    assert payload["name"] == "realworld-like-v1"
    assert payload["splits"]["test"]
    assert len(payload["splits"]["test"]) == 5

    assert "Generator: realworld_like_v1" in captured.out
