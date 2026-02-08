from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "diagram2code", *args],
        check=False,
        text=True,
        capture_output=True,
    )


def test_cli_dataset_list_smoke() -> None:
    p = _run("dataset", "list")
    assert p.returncode == 0
    assert "flowlearn" in p.stdout.splitlines()


def test_cli_dataset_info_smoke() -> None:
    p = _run("dataset", "info", "flowlearn")
    assert p.returncode == 0
    data = json.loads(p.stdout)
    assert data["name"] == "flowlearn"
    assert "version" in data
    assert "installed" in data


def test_cli_dataset_path_fails_when_not_installed(tmp_path: Path) -> None:
    p = _run("dataset", "path", "flowlearn", "--cache-dir", str(tmp_path))
    assert p.returncode == 2
    assert "Dataset not installed" in (p.stdout + p.stderr)


def test_cli_dataset_verify_fails_when_not_installed(tmp_path: Path) -> None:
    p = _run("dataset", "verify", "flowlearn", "--cache-dir", str(tmp_path))
    assert p.returncode == 2
