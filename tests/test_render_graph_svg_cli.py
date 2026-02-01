import subprocess
from pathlib import Path
import sys


def test_render_graph_svg_creates_file(tmp_path):
    img = Path("tests/fixtures/simple.png").resolve()
    export_dir = tmp_path / "bundle"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "diagram2code",
            str(img),
            "--export",
            str(export_dir),
            "--render-graph",
            "--render-format",
            "svg",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    out = export_dir / "graph.svg"
    assert out.exists()
    assert out.stat().st_size > 0
