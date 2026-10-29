from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
tree = ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))
extract_node = next(
    node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "extract_prompt"
)
namespace = {}
exec(compile(ast.Module(body=[extract_node], type_ignores=[]), "main.py", "exec"), namespace)
extract_prompt = namespace["extract_prompt"]


def test_extract_prompt():
    assert extract_prompt("/生图 一只猫") == "一只猫"
    assert extract_prompt("/生图 一只会戴帽子的猫") == "一只会戴帽子的猫"
    assert extract_prompt("一只猫") == "一只猫"
    assert extract_prompt("/生图") == ""
