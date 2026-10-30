from __future__ import annotations

import ast
from pathlib import Path

import pytest


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
    assert extract_prompt("/画图 一只会飞的猫", "画图") == "一只会飞的猫"
    assert extract_prompt("/生图 一只会飞的猫", "画图") == "一只会飞的猫"
    assert extract_prompt("一只猫") == "一只猫"
    assert extract_prompt("/生图") == ""


@pytest.mark.parametrize(
    ("message", "command", "expected"),
    [
        ("生图", "生图", ""),
        ("生图 一只猫", "生图", "一只猫"),
        ("生图\t一只猫", "生图", "一只猫"),
        ("生图\n一只猫", "生图", "一只猫"),
        ("/生图\t一只猫", "生图", "一只猫"),
        ("/生图\n一只猫", "生图", "一只猫"),
        ("画图", "画图", ""),
        ("/画图", "画图", ""),
        ("画图 一只猫", "画图", "一只猫"),
        ("生图 一只猫", "画图", "一只猫"),
        ("/画图\t一只猫", "画图", "一只猫"),
        ("画图\n一只猫", "画图", "一只猫"),
        ("生图片段", "生图", "生图片段"),
        ("/生图片段", "生图", "/生图片段"),
        ("画图说明", "画图", "画图说明"),
        ("生图 生图风格的猫", "生图", "生图风格的猫"),
    ],
)
def test_extract_prompt_framework_forms(message, command, expected):
    assert extract_prompt(message, command) == expected
