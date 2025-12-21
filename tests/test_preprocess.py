from pathlib import Path

from diagram2code.vision.preprocess import preprocess_image


def test_preprocess_writes_output(tmp_path: Path):
    fixture = Path("tests/fixtures/simple.png")
    assert fixture.exists(), "Add a test image at tests/fixtures/simple.png"

    result = preprocess_image(fixture, tmp_path)
    assert result.output_path.exists()
    assert result.output_path.stat().st_size > 0
