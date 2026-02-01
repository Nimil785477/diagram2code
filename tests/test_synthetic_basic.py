from __future__ import annotations

from pathlib import Path

from diagram2code.benchmark.dataset import load_dataset
from diagram2code.benchmark.predictor import PredGraph
from diagram2code.benchmark.runner import run_benchmark
from diagram2code.benchmark.synthetic_basic import generate_synthetic_basic


def test_generate_synthetic_basic_and_run_benchmark(tmp_path: Path):
    ds = tmp_path / "synthetic_basic"
    generate_synthetic_basic(ds, n=3)

    dataset = load_dataset(ds)

    # Oracle predictor: returns exact GT nodes/edges but with "pred ids" (strings)
    # so node matching + projection logic is exercised.
    gt_by_image = {item.image_path.name: item.gt for item in dataset.items}

    def predictor(image_path: Path) -> PredGraph:
        gt = gt_by_image[image_path.name]

        pred_nodes = []
        for n in gt.nodes:
            pred_nodes.append({"id": f"p{n.id}", "bbox": list(n.bbox)})

        pred_edges = []
        for e in gt.edges:
            pred_edges.append({"from": f"p{e.from_id}", "to": f"p{e.to_id}"})

        return PredGraph(nodes=pred_nodes, edges=pred_edges)

    result = run_benchmark(
        dataset_path=ds,
        predictor=predictor,
        alpha=0.35,
    )

    assert len(result.samples) == 3

    # Perfect prediction => perfect scores
    assert result.aggregate.node.f1 == 1.0
    assert result.aggregate.edge.f1 == 1.0
    assert result.aggregate.direction_accuracy == 1.0
    assert result.aggregate.exact_match_rate == 1.0
