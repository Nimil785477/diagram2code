from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _toposort(nodes: List[int], edges: List[Tuple[int, int]]) -> List[int]:
    # Kahn's algorithm (stable)
    outgoing: Dict[int, List[int]] = {n: [] for n in nodes}
    indeg: Dict[int, int] = {n: 0 for n in nodes}

    for a, b in edges:
        if a not in outgoing:
            outgoing[a] = []
            indeg[a] = 0
        if b not in outgoing:
            outgoing[b] = []
            indeg[b] = 0
        outgoing[a].append(b)
        indeg[b] += 1

    queue = [n for n in nodes if indeg.get(n, 0) == 0]
    queue.sort()

    order: List[int] = []
    while queue:
        n = queue.pop(0)
        order.append(n)
        for m in outgoing.get(n, []):
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
                queue.sort()

    # If cycle exists, fall back to node id order (don’t crash)
    if len(order) != len(nodes):
        return sorted(nodes)
    return order


def generate_program(graph: Dict[str, Any], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    nodes = [n["id"] for n in graph["nodes"]]
    edges = [(e["from"], e["to"]) for e in graph["edges"]]

    order = _toposort(nodes, edges)

    lines: List[str] = []
    lines.append('"""Auto-generated from diagram2code."""')
    lines.append("")
    for nid in order:
        lines.append(f"def node_{nid}():")
        lines.append(f"    print('Node {nid} executed')")
        lines.append("")
        lines.append("")

    lines.append("def main():")
    for nid in order:
        lines.append(f"    node_{nid}()")
    lines.append("")
    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    main()")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def generate_from_graph_json(graph_json_path: str | Path, out_path: str | Path) -> Path:
    graph_json_path = Path(graph_json_path)
    graph = json.loads(graph_json_path.read_text(encoding="utf-8"))
    return generate_program(graph, out_path)
