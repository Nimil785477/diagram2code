from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_cli_benchmark_info_prints_summary(tmp_path: Path) -> None:
    p = tmp_path / "result.json"
    p.write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "dataset": "example:minimal_v1",
                "split": "test",
                "predictor": "oracle",
                "num_samples": 3,
                "metrics": {"node_f1": 1.0, "edge_f1": 1.0, "exact_match_rate": 1.0},
                "run": {"timestamp_utc": "2026-02-12T00:00:00Z"},
            }
        ),
        encoding="utf-8",
    )

    cp = subprocess.run(
        [sys.executable, "-m", "diagram2code", "benchmark", "info", str(p)],
        capture_output=True,
        text=True,
    )
    assert cp.returncode == 0
    out = cp.stdout
    assert "schema_version:" in out
    assert "dataset:" in out
    assert "metrics:" in out
    assert "node_f1:" in out


def test_cli_benchmark_info_missing_file_errors(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"
    cp = subprocess.run(
        [sys.executable, "-m", "diagram2code", "benchmark", "info", str(missing)],
        capture_output=True,
        text=True,
    )
    assert cp.returncode != 0
    assert "Error:" in (cp.stdout + cp.stderr)
