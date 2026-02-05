from diagram2code.benchmark.predictor_backends import available_predictors


def test_available_predictors_includes_expected():
    preds = available_predictors()
    assert "oracle" in preds
    assert "heuristic" in preds
    assert "vision" in preds
