import subprocess
import sys
from pathlib import Path


def test_no_debug_suppresses_debug_artifacts(tmp_path):
    img = Path("tests/fixtures/simple.png").resolve()
    out_dir = tmp_path / "outputs"
    export_dir = tmp_path / "bundle"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "diagram2code",
            str(img),
            "--out",
            str(out_dir),
            "--export",
            str(export_dir),
            "--no-debug",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0

    # Essentials
    assert (out_dir / "graph.json").exists()
    assert (out_dir / "generated_program.py").exists()

    # Debug artifacts suppressed
    assert not (out_dir / "preprocessed.png").exists()
    assert not (out_dir / "debug_nodes.png").exists()
    assert not (out_dir / "debug_arrows.png").exists()
    assert not (out_dir / "render_graph.py").exists()

    # Export essentials
    assert (export_dir / "graph.json").exists()
    assert (export_dir / "generated_program.py").exists()
