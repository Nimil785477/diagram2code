from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_cli_print_graph_normal_run(tmp_path: Path):
    out_dir = tmp_path / "out"
    img = Path("tests/fixtures/branching.png")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "diagram2code.cli",
            str(img),
            "--out",
            str(out_dir),
            "--print-graph",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    # sanity: it printed the summary
    assert "Graph summary" in result.stdout
    assert "Nodes:" in result.stdout
    assert "Edges:" in result.stdout

    # normal run writes artifacts
    assert (out_dir / "graph.json").exists()
    assert (out_dir / "generated_program.py").exists()


def test_cli_print_graph_dry_run_does_not_write(tmp_path: Path):
    out_dir = tmp_path / "out"
    export_dir = tmp_path / "export"
    img = Path("tests/fixtures/branching.png")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "diagram2code.cli",
            str(img),
            "--out",
            str(out_dir),
            "--export",
            str(export_dir),
            "--dry-run",
            "--print-graph",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Dry run: no files will be written." in result.stdout
    assert "Graph summary" in result.stdout

    # dry-run must not create out/export dirs
    assert not out_dir.exists()
    assert not export_dir.exists()
