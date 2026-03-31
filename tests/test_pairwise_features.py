import math

from diagram2code.predictors.pairwise_features import extract_pair_features, feature_names


def _node(node_id: str, x: float, y: float, w: float = 40, h: float = 20) -> dict:
    return {"id": node_id, "bbox": [x, y, w, h]}


def test_extract_pair_features_is_deterministic():
    u = _node("u", 10, 10)
    v = _node("v", 100, 10)
    w = _node("w", 10, 100)

    feats1 = extract_pair_features(u, v, 400, 300, [u, v, w])
    feats2 = extract_pair_features(u, v, 400, 300, [u, v, w])

    assert feats1 == feats2
    assert len(feats1) == len(feature_names())


def test_extract_pair_features_direction_flags():
    u = _node("u", 10, 10)
    v = _node("v", 100, 10)
    w = _node("w", 10, 100)

    right_feats = extract_pair_features(u, v, 400, 300, [u, v, w])
    down_feats = extract_pair_features(u, w, 400, 300, [u, v, w])

    names = feature_names()
    by_name_right = dict(zip(names, right_feats, strict=True))
    by_name_down = dict(zip(names, down_feats, strict=True))

    assert by_name_right["is_right_of"] == 1.0
    assert by_name_right["is_left_of"] == 0.0
    assert by_name_down["is_below"] == 1.0
    assert by_name_down["is_above"] == 0.0


def test_extract_pair_features_values_are_finite():
    u = _node("u", 10, 10)
    v = _node("v", 100, 10)
    feats = extract_pair_features(u, v, 400, 300, [u, v])

    assert all(math.isfinite(x) for x in feats)
