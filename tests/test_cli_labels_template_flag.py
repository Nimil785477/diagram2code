from __future__ import annotations

import json
from pathlib import Path

from diagram2code.cli import main


def test_cli_writes_labels_template(tmp_path: Path):
    out_dir = tmp_path / "out"
    rc = main(
        [
            "tests/fixtures/simple.png",
            "--out",
            str(out_dir),
            "--labels-template",
        ]
    )
    assert rc == 0

    template_path = out_dir / "labels.template.json"
    assert template_path.exists()

    data = json.loads(template_path.read_text(encoding="utf-8"))

    # For simple.png we detect 2 nodes (your pipeline expectation)
    assert set(data.keys()) == {"0", "1"}
    assert data["0"] == ""
    assert data["1"] == ""
