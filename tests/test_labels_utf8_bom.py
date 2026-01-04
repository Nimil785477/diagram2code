from __future__ import annotations

import json
from pathlib import Path

from diagram2code.labels import load_labels


def test_load_labels_accepts_utf8_bom(tmp_path: Path):
    p = tmp_path / "labels.json"

    # Write JSON with UTF-8 BOM
    data = {"0": "Start", "1": "Left"}
    p.write_bytes(b"\xef\xbb\xbf" + json.dumps(data).encode("utf-8"))

    labels = load_labels(p)
    assert labels[0] == "Start"
    assert labels[1] == "Left"
