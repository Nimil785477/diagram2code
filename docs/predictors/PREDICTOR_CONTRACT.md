# Dataset Adapter Authoring Guide — diagram2code

**Status:** DRAFT (to be frozen in Phase 4)  
**Applies from:** v0.1.5+  
**Audience:** Predictor authors, benchmark developers

This document defines the **authoritative contract for predictors** in
`diagram2code`.

A predictor is a component that takes a single dataset sample as input and
produces a **predicted graph structure** suitable for benchmarking.

---

## 1. What is Predictor?
A predictor is a pure inference component that:
- Consumes a single `DatasetSample`
- Produces a predicted graph structure
- Does not access ground-truth data
- Does not perform evaluation or scoring

Predictors are **first-class, swappable components** in the benchmarking
pipeline.

## 2. Predictor Scope

Predictors may:
- Use image data from the sample
- Use metadata attached to the sample
- Load external models or weights
- Maintain internal state if necessary

Predictors must not:
- Read ground-truth graph JSON
- Modify dataset files
- Perform metric computation
- Perform file I/O for outputs

## 3. Predictor Interface
Each predictor MUST implement the following interface.
```python
class Predictor(Protocol):
    name: str

    def predict(
        self,
        sample: DatasetSample,
    ) -> GraphPrediction:
        ...
```
Interface Rules
- `predict()` MUST be deterministic for a fixed predictor state
- `predict()` MUST return a valid `GraphPrediction`
- Predictors MUST NOT raise on valid samples
- Errors should be surfaced during predictor initialization

## 4. Predictor Identification
Each predictor MUST define a stable name:
```python
name: str
```
Rules:
- Lowercase
- Alphanumeric plus hyphen or underscore
- Stable across releases

Example names:
- `oracle`
- `heuristic`
- `model_resnet50`

## 5. `GraphPrediction` Structure
A predictor returns a `GraphPrediction` object.
```
{
  "nodes": [...],
  "edges": [...]
}
```
Both fields MUST be present.

## 6. Predicted Nodes

Each predicted node MUST follow this structure:
```json
{
  "id": "node_1",
  "label": "optional text",
  "bbox": [x, y, w, h]
}
```
Rules:
- `id` MUST be unique within the prediction
- `bbox` format follows Phase-3 dataset contract
- `label` is optional

Predictors MAY:
- Reuse dataset node IDs
- Generate new node IDs
- Omit labels entirely

## 7. Predicted Edges

Each predicted edge MUST follow this structure:
```json
{
  "source": "node_1",
  "target": "node_2",
  "direction": "down"
}
```

Rules:
- `source` and `target` MUST reference predicted node IDs
- Self-loops are allowed
- Edge list MAY be empty

The `direction` field:
- Is optional
- Is evaluated separately
- Must not affect edge existence scoring

## 8. Empty Predictions
The following predictions are valid:
- Graphs with zero edges
- Graphs with zero nodes
- Graphs with nodes but no edges

Predictors MUST handle these cases gracefully.

## 9. Determinism Requirements
For benchmarking purposes:
- Predictors SHOULD be deterministic by default
- Any stochastic behavior MUST be explicitly documented
- Random seeds SHOULD be configurable at initialization

## 10. Predictor Lifecycle
Typical lifecycle:
1. Predictor is instantiated
2. Optional model or resource loading occurs
3. `predict(sample)` is called repeatedly
4. Predictor is discarded after benchmarking
Predictors SHOULD avoid per-sample expensive setup.

## 11. Registry Integration
Predictors MUST be registered in the predictor registry.

Example:
```python
PREDICTOR_REGISTRY = {
    "oracle": OraclePredictor,
    "heuristic": HeuristicPredictor,
}
```
Predictors are resolved via CLI:
```css
--predictor oracle
--predictor heuristic
--predictor model:/path/to/model
```
## 12. Evaluation Separation
Predictors MUST NOT:
- Compute precision, recall, or F1
- Access evaluation metrics
- Compare predictions to ground truth

All evaluation is handled by the benchmark runner.

## 13. Reference Predictors
The following predictors serve as references:
- `OraclePredictor` — upper-bound reference
- `HeuristicPredictor` — non-ML baseline

New predictors should be comparable against both.

## 14. Design Philosophy

Predictors should be:
- Simple
- Explicit
- Stateless when possible
- Easy to benchmark and compare

Predictors are **replaceable components**, not core infrastructure.
Any predictor that violates this contract is considered non-compliant.