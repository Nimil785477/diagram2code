from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_matplotlib_script(graph: dict[str, Any], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    nodes = graph["nodes"]
    edges = graph["edges"]

    # Build script as text (keeps it simple + editable)
    lines: list[str] = []
    lines.append("import matplotlib.pyplot as plt")
    lines.append("from matplotlib.patches import Rectangle")
    lines.append("")
    lines.append("def main():")
    lines.append("    fig, ax = plt.subplots()")
    lines.append("")

    # Draw nodes
    for n in nodes:
        nid = n["id"]
        x, y, w, h = n["bbox"]
        lines.append(f"    ax.add_patch(Rectangle(({x}, {y}), {w}, {h}, fill=False, linewidth=2))")
        lines.append(f"    ax.text({x}+{w}/2, {y}+{h}/2, 'Node {nid}', ha='center', va='center')")

    lines.append("")

    # Draw edges as arrows between bbox centers
    for e in edges:
        a = e["from"]
        b = e["to"]

        # find bboxes
        lines.append(f"    # edge {a}->{b}")
        lines.append(f"    na = next(n for n in {nodes!r} if n['id']=={a})")
        lines.append(f"    nb = next(n for n in {nodes!r} if n['id']=={b})")
        lines.append("    ax_a, ay_a, aw_a, ah_a = na['bbox']")
        lines.append("    ax_b, ay_b, aw_b, ah_b = nb['bbox']")
        lines.append("    x1, y1 = ax_a + aw_a/2, ay_a + ah_a/2")
        lines.append("    x2, y2 = ax_b + aw_b/2, ay_b + ah_b/2")
        lines.append(
            "    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), "
            "arrowprops=dict(arrowstyle='->', lw=2))"
        )
        lines.append("")

    # Final formatting
    lines.append("    ax.set_aspect('equal')")

    # set limits from nodes
    xs = [n["bbox"][0] for n in nodes]
    ys = [n["bbox"][1] for n in nodes]
    x2s = [n["bbox"][0] + n["bbox"][2] for n in nodes]
    y2s = [n["bbox"][1] + n["bbox"][3] for n in nodes]
    pad = 30

    xmin = min(xs) - pad
    ymin = min(ys) - pad
    xmax = max(x2s) + pad
    ymax = max(y2s) + pad

    lines.append(f"    ax.set_xlim({xmin}, {xmax})")
    lines.append(f"    ax.set_ylim({ymax}, {ymin})  # invert y to match image coords")
    lines.append("    ax.axis('off')")
    lines.append("    fig.savefig('outputs/render_graph.png', dpi=200, bbox_inches='tight')")
    lines.append("    print('✅ Wrote: outputs/render_graph.png')")
    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    main()")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def generate_from_graph_json(graph_json_path: str | Path, script_out_path: str | Path) -> Path:
    graph_json_path = Path(graph_json_path)
    graph = json.loads(graph_json_path.read_text(encoding="utf-8"))
    return generate_matplotlib_script(graph, script_out_path)
