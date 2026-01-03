import json
from pathlib import Path
from diagram2code.cli import main

def test_labels_template_values_empty(tmp_path: Path):
    rc = main(["tests/fixtures/branching.png", "--out", str(tmp_path), "--labels-template"])
    assert rc == 0
    p = tmp_path / "labels.template.json"
    assert p.exists()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert len(data) > 0
    assert all(v == "" for v in data.values())
