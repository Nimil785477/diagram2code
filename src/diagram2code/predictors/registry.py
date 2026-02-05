from __future__ import annotations

from diagram2code.predictors.base import Predictor
from diagram2code.predictors.oracle import OraclePredictor

PREDICTOR_REGISTRY: dict[str, type[Predictor]] = {
    "oracle": OraclePredictor,
}


def get_predictor(name: str) -> type[Predictor]:
    try:
        return PREDICTOR_REGISTRY[name]
    except KeyError as e:
        available = ", ".join(sorted(PREDICTOR_REGISTRY))
        raise ValueError(f"Unknown predictor '{name}'. Available: {available}") from e
