import json
from pathlib import Path

from diagram2code.export_program import generate_from_graph_json


def test_exported_program_has_ctx_and_todo(tmp_path: Path):
    graph = {
        "nodes": [{"id": 0, "bbox": [0, 0, 1, 1]}, {"id": 1, "bbox": [0, 0, 1, 1]}],
        "edges": [{"from": 0, "to": 1}],
    }
    graph_json = tmp_path / "graph.json"
    graph_json.write_text(json.dumps(graph), encoding="utf-8")

    labels = {0: "Step_1_Load_Data", 1: "Step_2_Train_Model"}

    out = tmp_path / "generated_program.py"
    generate_from_graph_json(graph_json, out, labels=labels)

    txt = out.read_text(encoding="utf-8")

    assert "def Step_1_Load_Data(ctx):" in txt
    assert "def Step_2_Train_Model(ctx):" in txt
    assert "ctx = {}" in txt
    assert "Step_1_Load_Data(ctx)" in txt
    assert "Step_2_Train_Model(ctx)" in txt
    assert "# TODO: implement logic here" in txt
