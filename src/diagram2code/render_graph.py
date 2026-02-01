from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

LayoutMode = Literal["auto", "topdown", "spring"]


@dataclass(frozen=True)
class RenderOptions:
    output_path: Path
    title: str | None = None
    show_node_ids: bool = True
    dpi: int = 200
    layout: LayoutMode = "auto"


def _load_graph(graph_json_path: Path) -> dict[str, Any]:
    with graph_json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_nodes_and_edges(
    data: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Tries common shapes:
      - {"nodes":[...], "edges":[...]}
      - {"nodes":[...], "arrows":[...]}
      - {"graph":{"nodes":[...], "edges":[...]}}
    Falls back to empty lists if not found.
    """
    if "nodes" in data and ("edges" in data or "arrows" in data):
        nodes = data.get("nodes", []) or []
        edges = data.get("edges", data.get("arrows", [])) or []
        return nodes, edges

    if "graph" in data and isinstance(data["graph"], dict):
        g = data["graph"]
        nodes = g.get("nodes", []) or []
        edges = g.get("edges", g.get("arrows", [])) or []
        return nodes, edges

    return [], []


def _node_label(node: dict[str, Any], labels: dict[str, str] | None) -> str:
    # Try label in node first
    for key in ("label", "text", "name"):
        if isinstance(node.get(key), str) and node[key].strip():
            return node[key].strip()

    # Try lookup by id/key
    node_id = node.get("id") or node.get("key") or node.get("node_id")
    if labels and node_id is not None:
        return labels.get(str(node_id), str(node_id))

    return str(node_id) if node_id is not None else "node"


def _topdown_layout(G) -> dict[str, tuple[float, float]]:
    """
    Produce a simple top-down (layered) layout for DAGs.
    Falls back to spring layout if cycles exist.
    """
    import networkx as nx

    if not nx.is_directed_acyclic_graph(G):
        return nx.spring_layout(G, seed=42)

    layers = list(nx.topological_generations(G))
    pos: dict[str, tuple[float, float]] = {}

    # Space nodes horizontally per layer
    for li, layer in enumerate(layers):
        layer_nodes = list(layer)
        n = len(layer_nodes)
        # center around 0: e.g. for n=3 -> x in [-1,0,1]
        for xi, node in enumerate(layer_nodes):
            x = float(xi) - (n - 1) / 2.0
            pos[node] = (x, -float(li))  # downwards as layer index increases

    # For any nodes not included (shouldn't happen), place them with spring
    if len(pos) != len(G.nodes):
        fallback = nx.spring_layout(G, seed=42)
        for k, v in fallback.items():
            pos.setdefault(k, (float(v[0]), float(v[1])))

    return pos


def _choose_layout(G, mode: LayoutMode) -> dict[str, tuple[float, float]]:
    import networkx as nx

    if mode == "spring":
        return nx.spring_layout(G, seed=42)
    if mode == "topdown":
        return _topdown_layout(G)

    # auto: try topdown if DAG, else spring
    return _topdown_layout(G)


def render_graph(
    graph_json_path: Path,
    output_path: Path,
    labels_json_path: Path | None = None,
    options: RenderOptions | None = None,
) -> Path:
    """
    Renders graph.json to an image (PNG or SVG depending on output_path suffix).

    Requirements:
      - networkx
      - matplotlib

    Returns output_path.
    """
    # Local imports so base install can work without render deps until flag used.
    import matplotlib.pyplot as plt
    import networkx as nx

    data = _load_graph(graph_json_path)
    nodes, edges = _infer_nodes_and_edges(data)

    labels: dict[str, str] | None = None
    if labels_json_path and labels_json_path.exists():
        with labels_json_path.open("r", encoding="utf-8-sig") as f:
            labels = json.load(f)

    opts = options or RenderOptions(output_path=output_path)

    G = nx.DiGraph()

    # Add nodes
    for n in nodes:
        node_id = n.get("id") or n.get("key") or n.get("node_id") or str(len(G.nodes))
        lbl = _node_label(n, labels)
        G.add_node(str(node_id), label=lbl)

    # Add edges (support common edge schemas)
    for e in edges:
        src = e.get("from") or e.get("src") or e.get("source")
        dst = e.get("to") or e.get("dst") or e.get("target")
        if src is None or dst is None:
            # Some formats might store as {"u":..,"v":..}
            src = e.get("u")
            dst = e.get("v")
        if src is None or dst is None:
            continue
        G.add_edge(str(src), str(dst))

    # If graph.json is empty or unexpected schema
    if len(G.nodes) == 0:
        raise ValueError(f"No nodes found in {graph_json_path}. Cannot render graph.")

    # Layout
    pos = _choose_layout(G, opts.layout)

    # Labels
    node_labels: dict[str, str] = {}
    for node_id, attrs in G.nodes(data=True):
        lbl = attrs.get("label", node_id)
        if opts.show_node_ids and lbl != node_id:
            node_labels[node_id] = f"{lbl}\n[{node_id}]"
        else:
            node_labels[node_id] = str(lbl)

    plt.figure(figsize=(10, 6))
    if opts.title:
        plt.title(opts.title)

    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos, arrows=True)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

    plt.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # SVG doesn't use DPI in the same way PNG does; it's fine to pass dpi anyway.
    plt.savefig(output_path, dpi=opts.dpi, bbox_inches="tight")
    plt.close()

    return output_path
