from diagram2code.predictors.learned import LearnedPredictor


def test_learned_predictor_returns_valid_graph():
    predictor = LearnedPredictor()

    sample = {
        "nodes": [
            {"id": "a", "bbox": [10, 10, 40, 20]},
            {"id": "b", "bbox": [100, 10, 40, 20]},
            {"id": "c", "bbox": [10, 100, 40, 20]},
        ],
        "metadata": {
            "image_width": 400,
            "image_height": 300,
        },
    }

    out = predictor.predict(sample)

    assert "nodes" in out
    assert "edges" in out
    assert out["nodes"] == sample["nodes"]

    seen = set()
    for edge in out["edges"]:
        assert edge["from"] != edge["to"]
        key = (edge["from"], edge["to"])
        assert key not in seen
        seen.add(key)
