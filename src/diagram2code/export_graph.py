from __future__ import annotations

import json
from pathlib import Path

from diagram2code.schema import Node


def save_graph_json(nodes: list[Node], edges: list[tuple[int, int]], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "nodes": [{"id": n.id, "bbox": list(n.bbox)} for n in nodes],
        "edges": [{"from": a, "to": b} for a, b in edges],
    }

    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out_path
