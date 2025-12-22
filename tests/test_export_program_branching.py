import json
from pathlib import Path

from diagram2code.export_program import generate_from_graph_json


def test_generate_program_branching(tmp_path: Path):
    graph = {
        "nodes": [
            {"id": 0, "bbox": [0, 0, 1, 1]},
            {"id": 1, "bbox": [0, 0, 1, 1]},
            {"id": 2, "bbox": [0, 0, 1, 1]},
            {"id": 3, "bbox": [0, 0, 1, 1]},
        ],
        "edges": [
            {"from": 0, "to": 1},
            {"from": 0, "to": 2},
            {"from": 1, "to": 3},
            {"from": 2, "to": 3},
        ],
    }

    labels = {
        0: "Step_1_Start",
        1: "Step_2_Left",
        2: "Step_2_Right",
        3: "Step_3_End",
    }

    graph_json = tmp_path / "graph.json"
    graph_json.write_text(json.dumps(graph), encoding="utf-8")

    out = tmp_path / "generated_program.py"
    generate_from_graph_json(graph_json, out, labels=labels)

    txt = out.read_text(encoding="utf-8")

    # functions exist
    assert "def Step_1_Start" in txt
    assert "def Step_2_Left" in txt
    assert "def Step_2_Right" in txt
    assert "def Step_3_End" in txt

    # main includes all calls
    assert "ctx = {}" in txt
    assert "Step_1_Start(ctx)" in txt
    assert "Step_2_Left(ctx)" in txt
    assert "Step_2_Right(ctx)" in txt
    assert "Step_3_End(ctx)" in txt

