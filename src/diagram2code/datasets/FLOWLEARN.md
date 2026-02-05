# FlowLearn Dataset Adapter — diagram2code

**Status:** STABLE  
**Applies from:** v0.1.4+  
**Audience:** Dataset adapter users, maintainers

This document defines **how the FlowLearn dataset adapter works for diagram2code**
and how FlowLearn datasets are converted into **Phase-3 compliant datasets**.

The FlowLearn adapter serves as the **reference implementation** of a Phase-3
dataset adapter.

---

## 1. What is Flowlearn?
FlowLearn is a public dataset containing synthetic and semi-synthetic
flowchart-style diagrams paired with structural representations.

The FlowLearn adapter converts supported FlowLearn subsets into
**Phase-3 compliant datasets** suitable for benchmarking and evaluation.

## 2. Adapter Location & Naming
The FlowLearn adapter lives in:
```
src/diagram2code/datasets/adapters/flowlearn.py
```
The adapter is registered under the dataset name:
```nginx
flowlearn
```
Each FlowLearn subset is converted into a separate Phase-3 dataset.

## 3. Supported Subsets
The FlowLearn adapter supports the following subsets:
- `SimFlowchart` (legacy minimal subset)
- `SimFlowchart/char`
- `SimFlowchart/word`

Dataset names emitted:
- `SimFlowchart` → `flowlearn-simflowchart`
- `SimFlowchart/char` → `flowlearn_simflowchart_char`
- `SimFlowchart/word` → `flowlearn_simflowchart_word`

The legacy name `flowlearn-simflowchart` is preserved for test compatibility.

## 4. Unsupported Subsets
The following FlowLearn subsets are not supported:
- `SciFlowchart`

If an unsupported subset is requested:
- The adapter MUST explicitly reject it
- A clear error message MUST be raised
- Silent fallback is forbidden

Example:
```text
SciFlowchart is not supported by the FlowLearn adapter
```
## 5. Conversion Rules
**Node Conversion**
- Nodes are extracted from OCR metadata
- Bounding boxes are derived from OCR layout information
- Node IDs are generated deterministically
- Labels are included when available

**Edge Conversion**
- Edges are parsed from Mermaid flowchart definitions
- Source and target nodes are resolved by ID
- Edge direction is inferred when possible

## 6. Empty-Edge Graphs
Some FlowLearn records (especially in `SimFlowchart/word`) may produce
graphs with **zero edges**.

Example:
```json
{
  "nodes": [...],
  "edges": []
}
```
This behavior is intentional and valid.

Reasons:
- Reflects ambiguity or incompleteness in the source dataset
- Preserves benchmark honesty
- Predictors must handle this case correctly

## 7. Strict vs Non-Strict Mode

**Strict Mode (Default)**
- Any invalid record → abort conversion
- Empty-edge graphs are allowed
- Malformed Mermaid definitions cause failure

**Non-Strict Mode**
- Invalid records skipped
- Adapter MUST log or count skipped records
- Remaining dataset must still be valid

Strict mode is the default for benchmarking.

## 8. Output Format
The FlowLearn adapter emits a Phase-3 compliant dataset layout:
```pqsql
dataset.json
images/
graphs/
```
Each sample contains:
- One image file
- One graph JSON file

All outputs conform to the Phase-3 dataset contract.

## 9. Known Limitations
- Mermaid parsing may fail on malformed definitions
- Some word-level samples contain no valid edges
- OCR metadata quality directly affects node bounding boxes

These limitations originate from the source dataset and are not corrected
by the adapter.

## 10. Testing Coverage
The FlowLearn adapter is covered by tests verifying:
- Minimal successful conversion
- Strict mode failure behavior
- Non-strict skipping behavior
- Dataset naming correctness
- Sample load via `load_dataset()`
- End-to-end benchmark compatibility

## 11. Role in the Project
The FlowLearn adapter serves as:
- The canonical Phase-3 adapter
- A benchmark sanity-check dataset
- A template for future dataset adapters

New adapters should aim to meet the same standard.

## 12. Adapter Design Philosophy
The FlowLearn adapter is designed to be:
- Deterministic
- Explicit
- Minimal
- Fail-fast

The adapter prioritizes benchmark integrity over dataset size.
Any behavior not explicitly documented here is considered unsupported.