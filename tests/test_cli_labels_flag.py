from pathlib import Path

from diagram2code.cli import main


def test_cli_accepts_labels_flag(tmp_path: Path):
    # Create a minimal labels file (doesn't need to match real nodes here)
    labels = tmp_path / "labels.json"
    labels.write_text('{"0": "Step_1_Test"}', encoding="utf-8")

    # Use fixture image; just ensure it runs without crashing
    rc = main(["tests/fixtures/simple.png", "--out", str(tmp_path), "--labels", str(labels)])
    assert rc == 0
