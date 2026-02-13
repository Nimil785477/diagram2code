from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED_TOP_LEVEL = {
    "schema_version",
    "dataset",
    "split",
    "predictor",
    "num_samples",
    "metrics",
    "run",
}

REQUIRED_METRICS = {
    "node_precision",
    "node_recall",
    "node_f1",
    "edge_precision",
    "edge_recall",
    "edge_f1",
    "direction_accuracy",
    "exact_match_rate",
    # runtime_mean_s optional
}


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/ci_verify_benchmark.py <path-to-result.json>")
        return 2

    p = Path(sys.argv[1])
    data = json.loads(p.read_text(encoding="utf-8"))

    missing = sorted(REQUIRED_TOP_LEVEL - set(data.keys()))
    if missing:
        print(f"Missing required top-level keys: {missing}")
        return 1

    if data["schema_version"] != "1.1":
        print(f"Unexpected schema_version={data['schema_version']!r} (expected '1.1')")
        return 1

    metrics = data["metrics"]
    if not isinstance(metrics, dict):
        print("metrics must be an object")
        return 1

    missing_m = sorted(REQUIRED_METRICS - set(metrics.keys()))
    if missing_m:
        print(f"Missing required metrics keys: {missing_m}")
        return 1

    print("Benchmark JSON sanity check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
