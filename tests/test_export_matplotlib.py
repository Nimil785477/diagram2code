import json
from pathlib import Path

from diagram2code.export_matplotlib import generate_from_graph_json


def test_generate_matplotlib_script(tmp_path: Path):
    graph = {
        "nodes": [
            {"id": 0, "bbox": [50, 124, 76, 76]},
            {"id": 1, "bbox": [199, 124, 76, 76]},
        ],
        "edges": [{"from": 0, "to": 1}],
    }

    graph_json = tmp_path / "graph.json"
    graph_json.write_text(json.dumps(graph), encoding="utf-8")

    out_script = tmp_path / "render_graph.py"
    generate_from_graph_json(graph_json, out_script)

    assert out_script.exists()
    txt = out_script.read_text(encoding="utf-8")
    assert "matplotlib" in txt
    assert "Rectangle" in txt
