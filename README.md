# diagram2code

Convert flowchart-style diagrams into runnable Python programs.
This repo is being built step-by-step.

Given a diagram image containing:
- Rectangular nodes
- Arrow connections
- Optional labels

diagram2code:
1. Detects nodes and directed edges
2. Builds a graph
3. Generates a runnable Python program
4. Optionally exports a self-contained bundle

## Installation

```bash
git clone https://github.com/Nimil785477/diagram2code.git
cd diagram2code
pip install -e ".[dev]"


✔ Why: people won’t guess editable installs.

---

### **Quick Start (MOST IMPORTANT SECTION)**

```md
## Quick Start

```bash
python -m diagram2code.cli examples/simple/diagram.png --out outputs

This will generate:

* outputs/preprocessed.png

* outputs/debug_nodes.png

* outputs/debug_arrows.png

* outputs/graph.json

* outputs/generated_program.py


✔ This answers: *“What do I get?”*

---

### **Run the generated program**
```md
```bash
python outputs/generated_program.py


✔ Closes the loop: image → code → execution.

---

### **Using labels**
```md
## Using labels

Provide a labels JSON file to name nodes:

```json
{
  "0": "Load_Data",
  "1": "Train_Model"
}

python -m diagram2code.cli diagram.png --out outputs --labels labels.json


✔ Shows value beyond demo.

---

### **Export bundle (shareable)**
```md
## Export bundle

```bash
python -m diagram2code.cli diagram.png --out outputs --export export_bundle

This creates a portable folder containing:

* generated_program.py

* graph.json

* run.ps1 (Windows)

* run.sh (Linux / macOS)

* README_EXPORT.md


Run it:
- Windows: `.\run.ps1`
- Mac/Linux: `bash run.sh`

# Simple Example

This example demonstrates a minimal diagram converted to Python code.

## Run
```bash
python -m diagram2code.cli diagram.png --out outputs --labels labels.json

Output
* Two nodes
* One directed edge
* Runnable Python Program


✔ Examples dramatically increase repo credibility.

---

## 8.3 CLI `--help` polish (small but powerful)

### In `cli.py`, ensure argparse looks like this:

```python
parser.add_argument("image", help="Path to diagram image")
parser.add_argument("--out", required=True, help="Output directory")
parser.add_argument("--labels", help="Optional labels.json file")
parser.add_argument("--export", help="Export runnable bundle")

Run:
python -m diagram2code.cli --help

## Dev setup
```bash
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
