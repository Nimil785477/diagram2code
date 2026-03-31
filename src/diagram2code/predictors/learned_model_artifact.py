from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LearnedModelArtifact:
    schema_version: int
    model_type: str
    feature_names: list[str]
    coef: list[float]
    intercept: float
    threshold: float
    top_k: int

    @classmethod
    def from_dict(cls, data: dict) -> LearnedModelArtifact:
        return cls(
            schema_version=int(data["schema_version"]),
            model_type=str(data["model_type"]),
            feature_names=[str(x) for x in data["feature_names"]],
            coef=[float(x) for x in data["coef"]],
            intercept=float(data["intercept"]),
            threshold=float(data.get("threshold", 0.5)),
            top_k=int(data.get("top_k", 2)),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> LearnedModelArtifact:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)


class LearnedEdgeScorer:
    def __init__(self, artifact: LearnedModelArtifact) -> None:
        self._artifact = artifact

    @property
    def threshold(self) -> float:
        return self._artifact.threshold

    @property
    def top_k(self) -> int:
        return self._artifact.top_k

    @property
    def feature_names(self) -> list[str]:
        return self._artifact.feature_names.copy()

    def score(self, features: list[float]) -> float:
        if len(features) != len(self._artifact.coef):
            raise ValueError(
                f"feature length mismatch: got {len(features)} expected {len(self._artifact.coef)}"
            )

        logit = self._artifact.intercept
        for weight, value in zip(self._artifact.coef, features, strict=True):
            logit += weight * value

        # numerically stable sigmoid
        if logit >= 0:
            z = math.exp(-logit)
            return 1.0 / (1.0 + z)

        z = math.exp(logit)
        return z / (1.0 + z)
