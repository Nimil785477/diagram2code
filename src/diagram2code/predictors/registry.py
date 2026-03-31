from __future__ import annotations

from diagram2code.predictors.base import Predictor
from diagram2code.predictors.heuristic import HeuristicPredictor
from diagram2code.predictors.learned import LearnedPredictor
from diagram2code.predictors.naive import NaivePredictor
from diagram2code.predictors.oracle import OraclePredictor

PREDICTOR_REGISTRY: dict[str, type[Predictor]] = {
    "oracle": OraclePredictor,
    "heuristic": HeuristicPredictor,
    "naive": NaivePredictor,
    "learned": LearnedPredictor,
}


def get_predictor(name: str) -> type[Predictor]:
    try:
        return PREDICTOR_REGISTRY[name]
    except KeyError as e:
        available = ", ".join(sorted(PREDICTOR_REGISTRY.keys()))
        raise ValueError(f"Unknown predictor: {name!r}. Available: {available}") from e
