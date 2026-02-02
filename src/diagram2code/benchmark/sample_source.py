from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from diagram2code.datasets import DatasetRegistry


@dataclass(frozen=True)
class BenchmarkSample:
    sample_id: str
    image_path: Path
    graph_path: Path


def iter_dataset_samples(
    dataset_ref: str | Path,
    *,
    split: str | None = None,
    limit: int | None = None,
    validate_graphs: bool = True,
) -> Iterable[BenchmarkSample]:
    ds = DatasetRegistry().load(dataset_ref, validate_graphs=validate_graphs)

    # Prefer deterministic default split selection.
    if split is None:
        splits = ds.splits()
        split = "test" if "test" in splits else ("all" if "all" in splits else splits[0])

    items = ds.samples(split)
    count = 0
    for s in items:
        if limit is not None and count >= limit:
            break
        yield BenchmarkSample(
            sample_id=s.sample_id,
            image_path=s.image_path,
            graph_path=s.graph_path,
        )
        count += 1
