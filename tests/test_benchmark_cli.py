from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from diagram2code.benchmark.synthetic_basic import generate_synthetic_basic


def test_benchmark_cli_oracle_smoke(tmp_path: Path):
    ds = tmp_path / "synthetic_basic"
    generate_synthetic_basic(ds, n=2)

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
            "--json",
            str(out_json),
        ],
        capture_output=True,
        text=True,
    )

    assert r.returncode == 0, r.stderr
    assert "node:" in r.stdout
    assert out_json.exists()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["aggregate"]["node"]["f1"] == 1.0
    assert data["aggregate"]["edge"]["f1"] == 1.0
