from pathlib import Path

from diagram2code.export_matplotlib import generate_from_graph_json


def test_generate_matplotlib_script(tmp_path: Path):
    graph_json = Path("outputs/graph.json")
    assert graph_json.exists(), "Run the CLI once to generate outputs/graph.json"

    out_script = tmp_path / "render_graph.py"
    generate_from_graph_json(graph_json, out_script)

    assert out_script.exists()
    txt = out_script.read_text(encoding="utf-8")
    assert "matplotlib" in txt
    assert "Rectangle" in txt
