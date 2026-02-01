from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


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


def write_benchmark_json(result, path: Path) -> None:
    """
    Write benchmark results to JSON.

    Important: benchmark results may contain nested custom objects (e.g. SampleResult),
    Paths, and other non-JSON types. We sanitize recursively before dumping.
    """
    # Prefer explicit conversion if available
    if hasattr(result, "to_dict") and callable(result.to_dict):
        data = result.to_dict()
    elif hasattr(result, "as_dict") and callable(result.as_dict):
        data = result.as_dict()
    elif is_dataclass(result):
        data = asdict(result)
    else:
        data = getattr(result, "__dict__", {"result": str(result)})

    safe = _sanitize_for_json(data)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe, indent=2), encoding="utf-8")
