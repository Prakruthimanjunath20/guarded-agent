"""Append-only JSONL traces: every model call, tool call and guardrail decision."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class Trace:
    def __init__(self, directory: Path | None = None, run_id: str | None = None) -> None:
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.events: list[dict[str, Any]] = []
        self.path = directory / f"{self.run_id}.jsonl" if directory else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **data: Any) -> dict[str, Any]:
        record = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "event": event,
            **data,
        }
        self.events.append(record)
        if self.path:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return record


def load_trace(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def render(events: list[dict[str, Any]]) -> str:
    """A readable timeline of a run."""
    lines = []
    for e in events:
        kind = e["event"]
        if kind == "run_start":
            lines.append(f"▶ task: {e['task']}")
        elif kind == "model_call":
            tools = ", ".join(e.get("tool_calls", [])) or "-"
            lines.append(f"  model (step {e['step']}): stop={e['stop_reason']} tools=[{tools}]")
        elif kind == "tool_call":
            status = e["status"].upper()
            lines.append(f"    {status:8} {e['tool']}({json.dumps(e['arguments'])[:80]})")
            if e.get("result"):
                lines.append(f"             → {e['result'][:100]!r}")
        elif kind == "guardrail":
            lines.append(f"    ⚠ guardrail [{e['kind']}]: {e['detail']}")
        elif kind == "run_end":
            lines.append(f"■ {e['stopped']}: {e['answer'][:200]!r}")
    return "\n".join(lines)
