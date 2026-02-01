import subprocess
import sys
from pathlib import Path


def test_render_graph_creates_png(tmp_path):
    img = Path("tests/fixtures/simple.png").resolve()
    out = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "diagram2code",
            str(img),
            "--export",
            str(out),
            "--render-graph",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert (out / "graph.png").exists()
    assert (out / "graph.png").stat().st_size > 0
