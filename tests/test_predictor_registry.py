import pytest

from diagram2code.predictors.registry import get_predictor


def test_get_predictor_oracle_resolves():
    cls = get_predictor("oracle")
    assert cls.__name__ == "OraclePredictor"


def test_get_predictor_unknown_raises():
    with pytest.raises(ValueError):
        get_predictor("does-not-exist")
