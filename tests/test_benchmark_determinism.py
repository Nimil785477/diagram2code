from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_benchmark_json_is_deterministic(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DIAGRAM2CODE_BENCHMARK_TIMESTAMP_UTC", "1970-01-01T00:00:00Z")

    out1 = tmp_path / "r1.json"
    out2 = tmp_path / "r2.json"

    cmd = [
        sys.executable,
        "-m",
        "diagram2code",
        "benchmark",
        "--dataset",
        "example:minimal_v1",
        "--predictor",
        "oracle",
        "--json",
    ]

    subprocess.check_call(cmd + [str(out1)])
    subprocess.check_call(cmd + [str(out2)])

    b1 = out1.read_bytes()
    b2 = out2.read_bytes()

    assert b1 == b2
    json.loads(out1.read_text(encoding="utf-8"))
