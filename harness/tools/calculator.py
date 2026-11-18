"""Example tool: safe arithmetic.

Models are unreliable at exact arithmetic, so give them a calculator. This
evaluates the expression's syntax tree directly and only allows numbers and
math operators — never `eval()` on model output.
"""

from __future__ import annotations

import ast
import operator
from typing import Annotated

from pydantic import Field

from harness.tools import tool

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("exponent too large")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)[:60]}")


@tool
def calculator(
    expression: Annotated[str, Field(description="Arithmetic expression, e.g. '(2 + 3) * 4 ** 2'")],
) -> str:
    """Evaluate an arithmetic expression with + - * / // % ** and parentheses."""
    result = _eval(ast.parse(expression, mode="eval").body)
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)
