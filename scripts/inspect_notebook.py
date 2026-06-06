import json
import os

notebook_path = "c:/Users/Yash/OneDrive/Desktop/IMD_AMV_DOF/PI-RAFT-INSAT/notebooks/baseline.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Number of cells: {len(nb['cells'])}")
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        code_lines = cell["source"]
        first_line = code_lines[0].strip() if code_lines else "EMPTY"
        print(f"Cell {idx} (Code) | Lines: {len(code_lines)} | First line: {first_line}")
