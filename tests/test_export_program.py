import json
from pathlib import Path

from diagram2code.export_program import generate_from_graph_json


def test_generate_program(tmp_path: Path):
    graph = {
        "nodes": [{"id": 0, "bbox": [0, 0, 1, 1]}, {"id": 1, "bbox": [0, 0, 1, 1]}],
        "edges": [{"from": 0, "to": 1}],
    }
    graph_json = tmp_path / "graph.json"
    graph_json.write_text(json.dumps(graph), encoding="utf-8")

    out = tmp_path / "generated_program.py"
    generate_from_graph_json(graph_json, out)

    txt = out.read_text(encoding="utf-8")
    assert "def node_0" in txt
    assert "def node_1" in txt
    assert "def main" in txt
    assert "node_0()" in txt and "node_1()" in txt
