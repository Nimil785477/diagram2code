import json
from pathlib import Path

from diagram2code.export_graph import save_graph_json
from diagram2code.schema import Node


def test_save_graph_json(tmp_path: Path):
    nodes = [Node(0, (1, 2, 3, 4)), Node(1, (5, 6, 7, 8))]
    edges = [(0, 1)]
    out = save_graph_json(nodes, edges, tmp_path / "graph.json")

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["edges"] == [{"from": 0, "to": 1}]
    assert data["nodes"][0]["bbox"] == [1, 2, 3, 4]
