from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def deep_sort(obj: Any) -> Any:
    """Recursively sort dict keys and list elements where safe.

    - dicts: keys sorted
    - lists: keep order unless elements are dict-like with stable keys we can sort by
    """
    if isinstance(obj, Mapping):
        return {k: deep_sort(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        # If list elements are dicts with common stable keys, sort them deterministically.
        if all(isinstance(x, Mapping) for x in obj):
            # Best-effort stable key:
            def sort_key(d: Mapping[str, Any]) -> tuple:
                # Prefer ids/labels if present, fallback to full key set.
                for primary in ("id", "node_id", "edge_id", "label", "src_id", "dst_id"):
                    if primary in d:
                        return (str(d.get(primary)),)
                return tuple(sorted((str(k), str(d.get(k))) for k in d.keys()))

            return [deep_sort(x) for x in sorted(obj, key=sort_key)]
        return [deep_sort(x) for x in obj]
    return obj
