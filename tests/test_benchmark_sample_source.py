from __future__ import annotations

from diagram2code.benchmark.sample_source import iter_dataset_samples


def test_iter_dataset_samples_example_default_split() -> None:
    samples = list(iter_dataset_samples("example:minimal_v1"))
    assert len(samples) == 1
    assert samples[0].sample_id == "0001"
    assert samples[0].image_path.suffix.lower() == ".svg"
    assert samples[0].graph_path.name == "0001.json"


def test_iter_dataset_samples_example_limit() -> None:
    samples = list(iter_dataset_samples("example:minimal_v1", limit=0))
    assert samples == []
