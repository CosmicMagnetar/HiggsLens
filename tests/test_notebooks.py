"""Cheap CI guard: notebooks stay valid JSON with syntactically valid code cells."""

import ast
import json
from pathlib import Path


def test_notebook_json_and_code_syntax():
    notebooks = sorted(Path("notebooks").glob("*.ipynb"))
    assert notebooks
    for path in notebooks:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload.get("nbformat", 0) >= 4
        for cell in payload["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=str(path))
