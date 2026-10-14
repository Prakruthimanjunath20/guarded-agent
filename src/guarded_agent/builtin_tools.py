"""Example tools: a safe calculator, unit conversion, a sandboxed notes folder and a
mock email sender that needs approval."""

from __future__ import annotations

import ast
import json
import math
import operator
import re
from datetime import UTC, datetime
from pathlib import Path

from guarded_agent.tools import Tool, ToolError, tool

# --- calculator ------------------------------------------------------------------

_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}  # fmt: skip
_FUNCTIONS = {"sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ToolError("exponent too large")
        return _OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        return _FUNCTIONS[node.func.id](*(_evaluate(a) for a in node.args))
    raise ToolError(f"unsupported expression: {ast.dump(node)[:60]}")


@tool()
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. "(12.5 * 4) / sqrt(16)".
    Supports + - * / // % **, parentheses and sqrt, abs, round, min, max."""
    if len(expression) > 200:
        raise ToolError("expression too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"not a valid expression: {exc.msg}") from exc
    try:
        value = _evaluate(tree)
    except ZeroDivisionError as exc:
        raise ToolError("division by zero") from exc
    return str(round(value, 10))


# --- units -----------------------------------------------------------------------

_LINEAR = {
    "length": {"m": 1.0, "km": 1000.0, "cm": 0.01, "mm": 0.001, "mi": 1609.344, "ft": 0.3048,
               "in": 0.0254},
    "mass": {"kg": 1.0, "g": 0.001, "lb": 0.45359237, "oz": 0.028349523125},
}  # fmt: skip


@tool()
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between units of length (m, km, cm, mm, mi, ft, in), mass (kg, g, lb, oz)
    or temperature (c, f, k)."""
    f, t = from_unit.lower(), to_unit.lower()
    temps = {"c", "f", "k"}
    if f in temps and t in temps:
        celsius = {"c": value, "f": (value - 32) * 5 / 9, "k": value - 273.15}[f]
        result = {"c": celsius, "f": celsius * 9 / 5 + 32, "k": celsius + 273.15}[t]
        return f"{round(result, 4)} {to_unit}"
    for table in _LINEAR.values():
        if f in table and t in table:
            return f"{round(value * table[f] / table[t], 6)} {to_unit}"
    raise ToolError(f"cannot convert {from_unit} to {to_unit}")


# --- sandboxed notes -------------------------------------------------------------


def notes_tools(root: Path) -> list[Tool]:
    """Read-only access to text notes inside `root`, and nowhere else."""
    root = root.resolve()

    def inside(name: str) -> Path:
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ToolError(f"no note called {name!r}")
        return path

    @tool(name="search_notes")
    def search_notes(query: str, limit: int = 5) -> str:
        """Search the notes folder for lines containing every word of the query.
        Returns matching file names and lines."""
        words = [w.lower() for w in re.findall(r"\w+", query)]
        hits = []
        for path in sorted(root.rglob("*.md")) + sorted(root.rglob("*.txt")):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if words and all(w in line.lower() for w in words):
                    hits.append(f"{path.relative_to(root)}:{number}: {line.strip()}")
        return "\n".join(hits[: max(1, min(limit, 20))]) or "no matches"

    @tool(name="read_note")
    def read_note(name: str) -> str:
        """Read one note by the file name returned from search_notes."""
        return inside(name).read_text(encoding="utf-8")[:4000]

    return [search_notes, read_note]


# --- email (side effect) ---------------------------------------------------------


def email_tool(outbox: Path, allowed_domains: set[str]) -> Tool:
    """A mock email sender: writes to an outbox folder, and only to allowed domains."""

    @tool(side_effect=True, max_calls=3, name="send_email")
    def send_email(to: str, subject: str, body: str) -> str:
        """Send an email. Needs human approval. Only company addresses are allowed."""
        domain = to.rsplit("@", 1)[-1].lower()
        if "@" not in to or domain not in allowed_domains:
            raise ToolError(f"recipient domain {domain!r} is not allowed")
        outbox.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        message = {"to": to, "subject": subject, "body": body}
        (outbox / f"{stamp}.json").write_text(json.dumps(message, indent=2))
        return f"email queued to {to}"

    return send_email
