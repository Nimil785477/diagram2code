# Dataset Adapter Authoring Guide — diagram2code

**Status:** STABLE  
**Applies from:** v0.1.4+  
**Audience:** Dataset adapter authors, maintainers

This document defines **how to write dataset adapters** for `diagram2code`
that produce **Phase-3 compliant datasets**.

Adapters are responsible for converting **raw external datasets** into the
standard Phase-3 dataset format.

---

## 1. What Is a Dataset Adapter?

A dataset adapter is a **pure conversion layer** that:

- Reads raw dataset files
- Validates and normalizes data
- Emits a Phase-3 dataset directory:
```
dataset.json
images/
graphs/
```
Adapters **do not**:
- Perform model inference
- Modify predictors
- Implement benchmarks
- Perform evaluation

## 2. Adapter Location & Naming
Adapters MUST live in:
```
src/diagram2code/datasets/adapters/
```

File naming convention:
```
<dataset_name>.py
```
Example:`Flowlearn.py`

Each adapter handles **one dataset family** (with subsets if needed).

## 3. Adapter Entry Point

Each adapter MUST expose a **single public entry function**.

Recommended signature:

```python
def convert(
    raw_dataset_path: Path,
    output_path: Path,
    *,
    split: str | None = None,
    strict: bool = True,
    limit: int | None = None,
) -> None:
    ...
```
### Parameter Semantics

| Parameter          | Description                                  |
| ------------------ | -------------------------------------------- |
| `raw_dataset_path` | Path to original dataset                     |
| `output_path`      | Destination directory for Phase-3 dataset    |
| `split`            | Optional split selector                      |
| `strict`           | Fail on first invalid sample if True         |
| `limit`            | Optional max samples (for testing/debugging) |

## 4. Adapter Responsibilities
**Required Responsibilities**

An adapter MUST:
1. Create Phase-3 directory layout
2. Generate a valid `dataset.json`
3. Copy or convert images into `images/`
4. Emit graph JSON into `graphs/`
5. Ensure ID consistency across files
6. Validate graph structure

**Optional Responsibilities**

Adapters MAY:

- Normalize labels
- Infer missing directions
- Skip known-bad samples (non-strict mode)
- Support subset selection

## 5. `dataset.json` Generation

Adapters MUST generate dataset.json with:
- Stable dataset name
- Explicit version
- Split membership

Example:
```json
{
  "name": "flowlearn-simflowchart",
  "version": "1.0",
  "splits": {
    "test": ["0001", "0002"]
  }
}
```
### Rules
- Sample IDs must match filenames
- No sample may appear in multiple splits
- Ordering must be deterministic

## 6. Sample Conversion Rules
**Image Handling**
- Images must be copied or converted verbatim
- No resizing, cropping, or normalization
- Extension may change, but ID must not

**Graph Generation**
- Graph JSON MUST satisfy Phase-3 contract
- Node IDs must be unique
- Edges must reference valid node IDs
- Empty edge lists are allowed

## 7. Error Handling Policy
**Strict Mode (Default)**
- Any invalid sample → abort conversion
- No partial datasets emitted

**Non-Strict Mode**
- Invalid samples skipped
- Adapter MUST log or count skipped samples
- Remaining dataset must still be valid
Adapters MUST NOT silently ignore errors in strict mode.

## 8. Validation Checklist
Before emitting a dataset, adapters SHOULD verify:
- All declared samples exist on disk
- Image and graph filenames match sample IDs
- Graph JSON schema validity
- No duplicate sample IDs
- No dangling edge references

Adapters MAY rely on shared validation utilities if available.

## 9. Unsupported Subsets
If a raw dataset subset is not supported:
- Adapter MUST explicitly reject it
- Clear error message required
- Silent fallback is forbidden

Example:
```text
SciFlowchart is not supported by the FlowLearn adapter
```

## 10. Registry Integration
Adapters MUST be registered in the dataset registry.

Example:
```python
DATASET_ADAPTERS = {
    "flowlearn": convert,
}
```
CLI resolution:
```
--dataset flowlearn:<subset>
```

## 11. Testing Requirements
Each adapter MUST have tests covering:
- Minimal successful conversion
- Strict mode failure
- Non-strict skipping behavior
- dataset.json correctness
- Sample load via `load_dataset()`

Tests SHOULD:
- Use small synthetic or minimal fixtures
- Avoid large raw datasets

## 12. Reference Adapter
The FlowLearn adapter is the canonical reference:
```
src/diagram2code/datasets/adapters/flowlearn.py
```
It demonstrates:
- Subset handling
- Strict vs non-strict behavior
- Partial dataset skipping
- Phase-3 compliant output

## 13. Adapter Design Philosophy
Adapters should be:
- Deterministic
- Transparent
- Minimal
- Fail-fast

Adapters are infrastructure, not experiments.
Any adapter that violates this guide is considered non-compliant.