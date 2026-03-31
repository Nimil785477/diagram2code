import json
from pathlib import Path

from diagram2code.predictors.learned_model_artifact import (
    LearnedEdgeScorer,
    LearnedModelArtifact,
)
from diagram2code.predictors.pairwise_features import feature_names


def test_learned_model_artifact_roundtrip(tmp_path: Path):
    path = tmp_path / "model.json"

    payload = {
        "schema_version": 1,
        "model_type": "logistic_regression",
        "feature_names": feature_names(),
        "coef": [0.0] * len(feature_names()),
        "intercept": 0.0,
        "threshold": 0.5,
        "top_k": 2,
    }

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    artifact = LearnedModelArtifact.from_path(path)
    scorer = LearnedEdgeScorer(artifact)

    assert artifact.schema_version == 1
    assert artifact.model_type == "logistic_regression"
    assert scorer.threshold == 0.5
    assert scorer.top_k == 2
    assert len(scorer.feature_names) == len(feature_names())
