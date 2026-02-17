from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_benchmark_on_flowlearn_smoke_fixture(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DIAGRAM2CODE_BENCHMARK_TIMESTAMP_UTC", "1970-01-01T00:00:00Z")
    monkeypatch.setenv("DIAGRAM2CODE_SEED", "0")

    dataset_root = Path("tests/fixtures/flowlearn_smoke")
    assert dataset_root.exists()

    out = tmp_path / "flowlearn_smoke.json"

    cmd = [
        sys.executable,
        "-m",
        "diagram2code",
        "benchmark",
        "--dataset",
        str(dataset_root),
        "--predictor",
        "oracle",
        "--json",
        str(out),
    ]
    subprocess.check_call(cmd)

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == "1.1"
    assert data["predictor"] == "oracle"
