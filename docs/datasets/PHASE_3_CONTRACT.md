# Phase 3 Dataset Contract — diagram2code

**Status:** STABLE  
**Applies from:** v0.1.4+  
**Audience:** Dataset authors, adapter authors, benchmark developers

This document defines the **authoritative contract** for all datasets used by
`diagram2code` from Phase 3 onward.

Any dataset that conforms to this contract is considered **benchmark-valid**.

---

## 1. Design Goals

Phase-3 datasets are designed to be:

- **Model-agnostic** — no ML assumptions baked in
- **Benchmark-ready** — comparable across predictors
- **Deterministic** — same input → same evaluation
- **Strictly validated** — invalid data fails fast

The dataset layer is **read-only** from the perspective of predictors.

---

## 2. Dataset Directory Layout

A valid Phase-3 dataset MUST follow this structure:

```
<dataset_root>/
├── dataset.json
├── images/
│ ├── <sample_id>.<ext>
│ └── ...
└── graphs/
├── <sample_id>.json
└── ...
```

### Rules
- `dataset.json` is **mandatory**
- `images/` and `graphs/` are **mandatory**
- Filenames (without extension) MUST match `sample_id`
- No nested subdirectories inside `images/` or `graphs/`

---

## 3. dataset.json Schema

`dataset.json` defines metadata and split membership.

### Required Fields

```json
{
  "name": "dataset_name",
  "version": "1.0",
  "splits": {
    "train": ["id1", "id2"],
    "val": ["id3"],
    "test": ["id4"]
  }
}
```

### Field Semantics
| Field     | Type   | Description                                                       |
| --------- | ------ | ----------------------------------------------------------------- |
| `name`    | string | Dataset identifier (stable, lowercase, hyphen/underscore allowed) |
| `version` | string | Dataset version (semantic or dataset-specific)                    |
| `splits`  | object | Mapping of split name → list of sample IDs                        |

### Split Rules

- At least one split MUST exist
- Sample IDs MUST be unique across all splits
- Split names are free-form (`train`, `val`, `test` recommended)
- Order inside splits is preserved

## 4. Sample Identity
A sample is identified by a unique `sample_id`.

For each `sample_id`, the following files MUST exist:

```
images/<sample_id>.<ext>
graphs/<sample_id>.json
```
### Image Rules

- Any standard image format allowed (`.png`, .`jpg`, .`jpeg`, `.bmp`)
- Image is treated as opaque input (no assumptions by dataset layer)

## 5. Graph JSON Contract
Each graph JSON represents the ground-truth structure of a diagram.
```
{
  "nodes": [...],
  "edges": [...]
}
```
Both fields MUST exist.

### Nodes 
```
{
  "id": "node_1",
  "label": "optional text",
  "bbox": [x, y, w, h]
}
```
**Required**
- `id` (string, unique within graph)
- `bbox` (array of 4 numbers)

**Optional**
- `label` (string)

**Bounding Box**
- Format: `[x, y, width, height]`
- Coordinate system: image pixel space
- Origin: top-left corner
- Units: pixels

### Edges
```
{
  "source": "node_1",
  "target": "node_2",
  "direction": "down"
}
```
**Required**
- `source` (node id)
- `target` (node id)

**Optional**
- `direction` (string)

**Direction Semantics**
- Direction is informational
- Missing direction is allowed
- Direction accuracy is evaluated separately

## 6. Empty-Edge Graphs
Graphs with **zero edges** are **VALID**.
```
{
  "nodes": [...],
  "edges": []
}
```

**Rationale**
- Represents real dataset failures or ambiguities
- Preserved for benchmark honesty
- Predictors must handle this case

## 7. DatasetSample API Expectations
A conforming dataset MUST support:
```
ds.splits()                  # → list[str]
ds.samples(split)            # → list[DatasetSample]
len(ds)                      # → total sample count
sample.load_graph_json()     # → parsed graph dict
```

**Guarantees**
- `DatasetSample.image_path` exists
- `DatasetSample.graph_path` exists
- `load_graph_json()` returns validated JSON

## 8. Validation Rules
A dataset is **INVALID** if:
- Any declared sample is missing files
- Graph JSON violates schema
- Duplicate sample IDs exist
- Node IDs are duplicated within a graph
- Edges reference nonexistent nodes

Invalid datasets MUST fail during loading (strict by default).

## 9. Strict vs Non-Strict Mode
- Strict mode (default):
    - Any invalid sample → dataset load fails
- Non-strict mode (opt-in):
    - Invalid samples skipped
    - Must be explicitly requested by adapter or CLI

Strict mode is the benchmark default.

## 10. Versioning Guarantees
- Phase-3 contract is backward compatible
- Breaking changes require a new phase
- Datasets must declare their own `version`

## 11. What Phase 3 Does NOT Define
Phase 3 intentionally excludes:
- Model architectures
- Predictor internals
- Training pipelines
- Image preprocessing rules

These belong to Phase 4+.

## 12. Reference Implementations
- FlowLearn adapter: `datasets/adapters/flowlearn.py`
- Oracle predictor: benchmark reference
- Tests: `tests/test_*dataset*`
