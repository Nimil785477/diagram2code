# diagram2code

Convert simple flowchart-style diagrams into runnable Python programs.

`diagram2code` takes a diagram image (rectangular steps + arrows), detects the flow, and generates:
- a graph representation
- a runnable Python program
- optional debug visualizations
- an exportable bundle

This project is designed for **learning, prototyping, and experimentation**, not for production-grade diagram parsing.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Using Labels](#using-labels)
4. [Export Bundle](#export-bundle)
5. [Generated Files](#generated-files)
6. [Examples](#examples)
7. [Limitations](#limitations)

---

## Installation
Clone the repository and install in editable mode:

git clone https://github.com/Nimil785477/diagram2code.git
cd diagram2code

python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows
.\.venv\Scripts\activate

pip install -e .

## Quick Start
Run diagram2code on a simple diagram:

python -m diagram2code.cli examples/simple/diagram.png --out outputs

## Using Labels
You can provide custom labels for nodes using a JSON file

Example labels.json
```
{
  "0": "Start",
  "1": "Process",
  "2": "End"
}
```
Run with labels
```
python -m diagram2code.cli diagram.png --out outputs --labels labels.json
```

## Export Bundle
The --export flag creates a self-contained runnable bundle.
```
python -m diagram2code.cli diagram.png --out outputs --export export_bundle
```

When using --export, the following files are copied:
```
export_bundle/
├── generated_program.py
├── graph.json
├── labels.json            (if provided)
├── debug_nodes.png        (if exists)
├── debug_arrows.png       (if exists)
├── render_graph.py        (if exists)
├── run.ps1
├── run.sh
└── README_EXPORT.md
```

## Generated Files
After a normal run (--out outputs):
| File                   | Description                          |
| ---------------------- | ------------------------------------ |
| `preprocessed.png`     | Binary image used for detection      |
| `debug_nodes.png`      | Detected rectangles overlay          |
| `debug_arrows.png`     | Detected arrows overlay (if enabled) |
| `graph.json`           | Graph structure (nodes + edges)      |
| `render_graph.py`      | Script to visualize the graph        |
| `generated_program.py` | Generated executable Python program  |

## Examples
Simple linear flow
```
[ A ] → [ B ] → [ C ]
```
Branching flow
```
      → [ B ]
[ A ]
      → [ C ]
```
## Limitations
|Only rectangular nodes are supported
|Arrow detection is heuristic-based
|Complex curves, diagonals, or overlapping arrows may fail
|No text extraction from inside shapes
|Not intended for UML, BPMN, or hand-drawn diagrams





