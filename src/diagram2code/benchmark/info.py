from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class BenchmarkInfoError(RuntimeError):
    pass


def load_result_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise BenchmarkInfoError(f"Result file not found: {path}") from e
    except json.JSONDecodeError as e:
        raise BenchmarkInfoError(f"Invalid JSON: {path}") from e

    if not isinstance(data, dict):
        raise BenchmarkInfoError("Result JSON must be an object.")
    return data


def format_result_summary(data: dict[str, Any]) -> str:
    # Required-ish fields (we print 'unknown' instead of crashing)
    schema_version = data.get("schema_version", "unknown")
    dataset = data.get("dataset", "unknown")
    split = data.get("split", "unknown")
    predictor = data.get("predictor", "unknown")
    num_samples = data.get("num_samples", "unknown")

    metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
    run = data.get("run") if isinstance(data.get("run"), dict) else {}

    # Metrics of interest (only print if present)
    metric_keys = [
        "node_f1",
        "edge_f1",
        "direction_accuracy",
        "exact_match_rate",
        "runtime_mean_s",
    ]

    lines: list[str] = []
    lines.append(f"schema_version: {schema_version}")
    lines.append(f"dataset: {dataset}")
    lines.append(f"split: {split}")
    lines.append(f"predictor: {predictor}")
    lines.append(f"num_samples: {num_samples}")
    lines.append("")
    lines.append("metrics:")
    any_metric = False
    for k in metric_keys:
        if k in metrics:
            lines.append(f"  {k}: {metrics[k]}")
            any_metric = True
    if not any_metric:
        lines.append("  (none)")
    lines.append("")
    lines.append("run:")
    run_keys = ["timestamp_utc", "diagram2code_version", "git_sha"]
    any_run = False
    for k in run_keys:
        if k in run:
            lines.append(f"  {k}: {run[k]}")
            any_run = True
    if not any_run:
        lines.append("  (none)")

    return "\n".join(lines)
