from pathlib import Path
import json
import subprocess
import sys


def test_cli_export_bundle(tmp_path: Path):
    out_dir = tmp_path / "out"
    export_dir = tmp_path / "export"
    img = Path("tests/fixtures/simple.png")

    cmd = [
        sys.executable,
        "-m",
        "diagram2code.cli",
        str(img),
        "--out",
        str(out_dir),
        "--export",
        str(export_dir),
    ]
    subprocess.check_call(cmd)

    assert (export_dir / "graph.json").exists()
    assert (export_dir / "generated_program.py").exists()
    assert (export_dir / "README_EXPORT.md").exists()

    g = json.loads((export_dir / "graph.json").read_text(encoding="utf-8"))
    assert "nodes" in g and "edges" in g
