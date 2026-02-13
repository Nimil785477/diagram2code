from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from diagram2code.benchmark.result_schema import SCHEMA_VERSION, BenchmarkResult


def _sanitize_for_json(obj: Any) -> Any:
    """
    Recursively convert obj into JSON-serializable primitives:
    dict / list / str / int / float / bool / None.
    Handles Path, dataclasses, pydantic, and common custom objects.
    """
    # Fast path: primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # Path -> string
    if isinstance(obj, Path):
        return str(obj)

    # numpy scalars -> python scalars (optional)
    try:
        import numpy as np  # type: ignore

        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
    except Exception:
        pass

    # dict
    if isinstance(obj, dict):
        # ensure keys are strings (JSON requires that)
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}

    # list/tuple/set
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(x) for x in obj]

    # dataclass instance
    if is_dataclass(obj):
        return _sanitize_for_json(asdict(obj))

    # pydantic v2
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return _sanitize_for_json(obj.model_dump())

    # common "to dict" patterns
    for name in ("to_dict", "as_dict", "dict"):
        if hasattr(obj, name) and callable(getattr(obj, name)):
            try:
                return _sanitize_for_json(getattr(obj, name)())
            except TypeError:
                # some .dict() require kwargs; ignore and fall back
                pass

    # generic object -> __dict__ (best-effort)
    if hasattr(obj, "__dict__"):
        return _sanitize_for_json(obj.__dict__)

    # final fallback: string representation (keeps JSON writing from crashing)
    return str(obj)


def _diagram2code_version() -> str:
    try:
        from importlib.metadata import version

        return version("diagram2code")
    except Exception:
        return ""


def _git_sha_short() -> str:
    """
    Best-effort git sha from the current working tree.
    Returns empty string if not in a git repo or git is unavailable.
    """
    try:
        cp = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return cp.stdout.strip()
    except Exception:
        return ""


def _now_utc_iso() -> str:
    forced = os.environ.get("DIAGRAM2CODE_BENCHMARK_TIMESTAMP_UTC")
    if forced:
        return forced
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _get_attr_any(obj: Any, *names: str) -> Any:
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


def write_benchmark_json(
    result: Any,
    path: Path,
    *,
    extra_run_meta: dict[str, Any] | None = None,
) -> None:
    """
    Write benchmark results to JSON using the frozen schema v1 (BenchmarkResult).

    `extra_run_meta` is merged into the `run` block (keys with None values are ignored).
    This is the intended extension point for CLI-provided provenance like:
      - cli
      - dataset_ref
      - dataset_root
      - predictor_out
      - dataset_manifest_sha256
    """
    agg = _get_attr_any(result, "aggregate")

    # Best-effort stable identifiers
    dataset_id = _get_attr_any(result, "dataset", "dataset_ref", "dataset_path", "dataset_root")
    split = _get_attr_any(result, "split")
    predictor = _get_attr_any(result, "predictor", "predictor_name")

    dataset_str = str(dataset_id) if dataset_id is not None else "unknown"
    split_str = str(split) if split is not None else "unknown"
    predictor_str = str(predictor) if predictor is not None else "unknown"
    # Prefer explicit CLI provenance when provided (result object may not carry these)
    if extra_run_meta:
        if extra_run_meta.get("dataset_ref") is not None:
            dataset_str = str(extra_run_meta["dataset_ref"])
        if extra_run_meta.get("split") is not None:
            split_str = str(extra_run_meta["split"])
        if extra_run_meta.get("predictor") is not None:
            predictor_str = str(extra_run_meta["predictor"])
    # num_samples
    num_samples = _get_attr_any(result, "num_samples")
    if num_samples is None:
        samples = _get_attr_any(result, "samples", "per_sample")
        num_samples = len(samples) if samples is not None else 0

    # metrics
    metrics: dict[str, float] = {}
    if agg is not None:
        node = _get_attr_any(agg, "node")
        edge = _get_attr_any(agg, "edge")

        if node is not None:
            metrics["node_precision"] = _safe_float(_get_attr_any(node, "precision"))
            metrics["node_recall"] = _safe_float(_get_attr_any(node, "recall"))
            metrics["node_f1"] = _safe_float(_get_attr_any(node, "f1"))

        if edge is not None:
            metrics["edge_precision"] = _safe_float(_get_attr_any(edge, "precision"))
            metrics["edge_recall"] = _safe_float(_get_attr_any(edge, "recall"))
            metrics["edge_f1"] = _safe_float(_get_attr_any(edge, "f1"))

        metrics["direction_accuracy"] = _safe_float(_get_attr_any(agg, "direction_accuracy"))
        metrics["exact_match_rate"] = _safe_float(_get_attr_any(agg, "exact_match_rate"))

        runtime = _get_attr_any(agg, "runtime_mean_s")
        if runtime is not None:
            metrics["runtime_mean_s"] = _safe_float(runtime)

    run: dict[str, Any] = {
        "timestamp_utc": _now_utc_iso(),
        "diagram2code_version": _diagram2code_version(),
        "git_sha": _git_sha_short(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    if extra_run_meta:
        # ignore None values so callers can pass optional fields cleanly
        for k in sorted(extra_run_meta.keys()):
            v = extra_run_meta[k]
            if v is not None:
                run[k] = v

    out = BenchmarkResult(
        schema_version=SCHEMA_VERSION,
        dataset=dataset_str,
        split=split_str,
        predictor=predictor_str,
        num_samples=int(num_samples),
        metrics=metrics,
        run=_sanitize_for_json(run),
    )
    out.validate()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(out.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
