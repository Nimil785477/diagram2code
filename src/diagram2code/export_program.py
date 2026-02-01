from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from diagram2code.labels import to_valid_identifier


def _toposort(nodes: list[int], edges: list[tuple[int, int]]) -> list[int]:
    outgoing: dict[int, list[int]] = {n: [] for n in nodes}
    indeg: dict[int, int] = {n: 0 for n in nodes}

    for a, b in edges:
        outgoing.setdefault(a, [])
        outgoing.setdefault(b, [])
        indeg.setdefault(a, 0)
        indeg.setdefault(b, 0)
        outgoing[a].append(b)
        indeg[b] += 1

    queue = sorted([n for n in outgoing.keys() if indeg.get(n, 0) == 0])

    order: list[int] = []
    while queue:
        n = queue.pop(0)
        order.append(n)
        for m in outgoing.get(n, []):
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
                queue.sort()

    # If cycle or mismatch, fall back to stable ordering
    if len(order) != len(outgoing):
        return sorted(outgoing.keys())
    return order


def generate_program(
    graph: dict[str, Any],
    out_path: str | Path,
    labels: dict[int, str] | None = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = labels or {}

    def fname(nid: int) -> str:
        raw = labels.get(nid, f"node_{nid}")
        return to_valid_identifier(raw, fallback=f"node_{nid}")

    nodes = [n["id"] for n in graph["nodes"]]
    edges = [(e["from"], e["to"]) for e in graph["edges"]]
    order = _toposort(nodes, edges)

    lines: list[str] = []
    lines.append('"""Auto-generated from diagram2code."""')
    lines.append("")
    lines.append("# Each step receives a shared execution context dict named `ctx`.")
    lines.append("# Add your own state into `ctx` inside any step function.")
    lines.append("")

    # Define functions
    for nid in order:
        fn = fname(nid)
        lines.append(f"def {fn}(ctx):")
        lines.append("    # TODO: implement logic here")
        lines.append(f"    print('{fn} executed')")
        lines.append("")
        lines.append("")

    # main calls them in topological order
    lines.append("def main():")
    lines.append("    ctx = {}")
    for nid in order:
        lines.append(f"    {fname(nid)}(ctx)")
    lines.append("")
    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    main()")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def generate_from_graph_json(
    graph_json_path: str | Path,
    out_path: str | Path,
    labels: dict[int, str] | None = None,
) -> Path:
    graph_json_path = Path(graph_json_path)
    graph = json.loads(graph_json_path.read_text(encoding="utf-8"))
    return generate_program(graph, out_path, labels=labels)
