from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from diagram2code.datasets.synthetic_basic import generate_synthetic_basic


def test_benchmark_cli_oracle_smoke(tmp_path: Path) -> None:
    ds = tmp_path / "synthetic_basic"
    generate_synthetic_basic(ds, n=2, seed=0, split="test")

    out_json = tmp_path / "result.json"

    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "diagram2code.cli",
            "benchmark",
            "--dataset",
            str(ds),
            "--alpha",
            "0.35",
            "--predictor",
            "oracle",
            "--split",
            "test",
            "--limit",
            "2",
            "--json",
            str(out_json),
        ],
        capture_output=True,
        text=True,
    )

    assert r.returncode == 0, r.stderr
    assert out_json.exists()
