import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "win32",
    reason="PowerShell script test is Windows-only",
)


def test_export_bundle_run_ps1(tmp_path: Path):
    out_dir = tmp_path / "out"
    export_dir = tmp_path / "export"
    img = Path("tests/fixtures/branching.png")

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "diagram2code.cli",
            str(img),
            "--out",
            str(out_dir),
            "--export",
            str(export_dir),
        ]
    )

    ps1 = export_dir / "run.ps1"
    assert ps1.exists()

    subprocess.check_call(
        ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(ps1)], cwd=str(tmp_path)
    )
