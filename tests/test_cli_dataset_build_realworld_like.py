from __future__ import annotations

import json
from pathlib import Path

from diagram2code.cli import main


def test_cli_dataset_build_realworld_like_includes_expected_motif_metadata(tmp_path) -> None:
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
            "8",
            "--seed",
            "0",
        ]
    )

    assert rc == 0

    payload = json.loads((out_dir / "dataset.json").read_text(encoding="utf-8"))
    motifs = payload["generator"]["motifs"]

    assert "simple_horizontal" in motifs
    assert "branch_merge" in motifs
    assert "staged_directional" in motifs
    assert "fan_out_pipeline" in motifs
    assert "staggered_multirow" in motifs
    assert "loop_return" in motifs


def test_cli_dataset_build_realworld_like_writes_matching_image_and_graph_counts(
    tmp_path: Path,
) -> None:
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
            "8",
            "--seed",
            "0",
        ]
    )

    assert rc == 0

    images = sorted((out_dir / "images").glob("*.png"))
    graphs = sorted((out_dir / "graphs").glob("*.json"))

    assert len(images) == 8
    assert len(graphs) == 8
