from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from diagram2code.benchmark.result_schema import SCHEMA_VERSION, BenchmarkResult

# Keep in sync with docs/benchmarks/LEADERBOARD_FORMAT.md
CSV_COLUMNS: list[str] = [
    "timestamp_utc",
    "dataset",
    "split",
    "predictor",
    "schema_version",
    "num_samples",
    "exact_match_rate",
    "edge_f1",
    "node_f1",
    "direction_accuracy",
    "runtime_mean_s",
    "diagram2code_version",
    "git_sha",
    "platform",
    "python",
]


@dataclass(frozen=True)
class LeaderboardRow:
    data: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        # Ensure all known columns exist (missing -> "")
        out: dict[str, Any] = {}
        for k in CSV_COLUMNS:
            out[k] = self.data.get(k, "")
        # Allow extra keys if ever added later
        for k, v in self.data.items():
            if k not in out:
                out[k] = v
        return out


def _coerce_result_dict(raw: dict[str, Any]) -> dict[str, Any]:
    # Make loads robust across older JSONs (best-effort).
    raw = dict(raw)
    raw.setdefault("schema_version", SCHEMA_VERSION)
    raw.setdefault("metrics", {})
    raw.setdefault("run", {})
    return raw


def load_result(path: Path) -> BenchmarkResult:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Result JSON must be an object: {path}")

    # Must look like our benchmark schema
    required = {"schema_version", "dataset", "split", "predictor", "num_samples", "metrics"}
    if not required.issubset(raw.keys()):
        missing = sorted(required - set(raw.keys()))
        raise ValueError(f"Not a benchmark result JSON (missing {missing}): {path}")

    data = _coerce_result_dict(raw)
    r = BenchmarkResult(**data)
    r.validate()
    return r


def result_to_row(r: BenchmarkResult) -> LeaderboardRow:
    m = dict(r.metrics or {})
    run = dict(r.run or {})

    row: dict[str, Any] = {
        "timestamp_utc": run.get("timestamp_utc", ""),
        "dataset": r.dataset,
        "split": r.split,
        "predictor": r.predictor,
        "schema_version": r.schema_version,
        "num_samples": r.num_samples,
        "exact_match_rate": m.get("exact_match_rate", ""),
        "edge_f1": m.get("edge_f1", ""),
        "node_f1": m.get("node_f1", ""),
        "direction_accuracy": m.get("direction_accuracy", ""),
        "runtime_mean_s": m.get("runtime_mean_s", ""),
        "diagram2code_version": run.get("diagram2code_version", ""),
        "git_sha": run.get("git_sha", ""),
        "platform": run.get("platform", ""),
        "python": run.get("python", ""),
    }
    return LeaderboardRow(row)


def build_rows(paths: Iterable[Path]) -> list[LeaderboardRow]:
    rows: list[LeaderboardRow] = []
    skipped: list[str] = []

    for p in paths:
        try:
            r = load_result(p)
        except Exception as e:
            skipped.append(f"{p}: {e}")
            continue
        rows.append(result_to_row(r))

    if skipped and not rows:
        # If nothing valid was found, fail loudly.
        details = "\n".join(f"  - {s}" for s in skipped[:20])
        raise ValueError(
            f"No valid benchmark result JSON files found.\nExamples of skipped files:\n{details}"
        )

    return rows


def write_csv(rows: Iterable[LeaderboardRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow(row.as_dict())


def write_md(rows: Iterable[LeaderboardRow], out_path: Path) -> None:
    # Simple Markdown table (no deps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = [r.as_dict() for r in rows]

    def cell(v: Any) -> str:
        s = "" if v is None else str(v)
        return s.replace("\n", " ").replace("|", "\\|")

    header = "| " + " | ".join(CSV_COLUMNS) + " |\n"
    sep = "| " + " | ".join(["---"] * len(CSV_COLUMNS)) + " |\n"
    body = ""
    for r in rows_list:
        body += "| " + " | ".join(cell(r.get(c, "")) for c in CSV_COLUMNS) + " |\n"

    out_path.write_text(header + sep + body, encoding="utf-8")
