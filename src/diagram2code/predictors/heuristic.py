from __future__ import annotations

import math

from diagram2code.datasets.types import DatasetSample
from diagram2code.predictors.types import GraphPrediction


class HeuristicPredictor:
    """
    Intermediate deterministic baseline predictor.

    Strategy:
    - Use dataset nodes directly
    - Normalize/sort nodes by geometric center
    - Infer forward edges using simple geometry:
      - prefer nodes to the right
      - also allow downward progression
      - allow up to 2 outgoing edges for likely split cases
    """

    name = "heuristic"

    def predict(self, sample: DatasetSample) -> GraphPrediction:
        gt = sample.load_graph_json()
        nodes = list(gt.get("nodes", []))

        norm_nodes: list[dict] = []
        for n in nodes:
            bbox = list(n.get("bbox", [0, 0, 0, 0]))
            if len(bbox) != 4:
                bbox = [0, 0, 0, 0]

            norm_nodes.append(
                {
                    "id": str(n.get("id")),
                    "bbox": bbox,
                }
            )

        def center(node: dict) -> tuple[float, float]:
            x, y, w, h = node["bbox"]
            return (x + w / 2, y + h / 2)

        def center_key(node: dict) -> tuple[float, float]:
            cx, cy = center(node)
            return (cy, cx)

        norm_nodes.sort(key=center_key)

        edges: list[dict] = []
        seen_edges: set[tuple[str, str]] = set()

        for src in norm_nodes:
            src_id = src["id"]
            src_cx, src_cy = center(src)

            candidates: list[tuple[float, float, float, str]] = []
            for dst in norm_nodes:
                dst_id = dst["id"]
                if dst_id == src_id:
                    continue

                dst_cx, dst_cy = center(dst)
                dx = dst_cx - src_cx
                dy = dst_cy - src_cy

                # only consider forward-ish moves:
                # - clearly rightward
                # - or strongly downward
                if dx < 20 and dy < 30:
                    continue

                dist = math.hypot(dx, dy)

                # direction preference:
                # favor rightward moves, but allow downward flow too
                right_bonus = 0.0 if dx >= 20 else 80.0
                upward_penalty = 120.0 if dy < -20 else 0.0

                score = dist + right_bonus + upward_penalty
                candidates.append((score, abs(dy), dx, dst_id))

            candidates.sort(key=lambda t: (t[0], t[1], -t[2], t[3]))

            if not candidates:
                continue

            # always keep the best forward candidate
            chosen = [candidates[0]]

            # optional second edge for likely split cases:
            # keep second candidate only if it is reasonably competitive
            # and vertically separated from the best candidate
            if len(candidates) >= 2:
                best = candidates[0]
                second = candidates[1]

                best_score, best_abs_dy, _, _ = best
                second_score, second_abs_dy, _, _ = second

                if second_score <= best_score + 80 and abs(second_abs_dy - best_abs_dy) >= 40:
                    chosen.append(second)

            for _, _, _, dst_id in chosen:
                key = (src_id, dst_id)
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                edges.append(
                    {
                        "source": src_id,
                        "target": dst_id,
                        "direction": "forward",
                    }
                )

        return {
            "nodes": norm_nodes,
            "edges": edges,
        }
