"""A stand-in for the Claude client that replays a fixed plan.

It returns objects shaped like Messages API responses (content blocks, stop_reason,
usage), so the real agent loop, tools and guardrails run unchanged. Used for the
offline demo and the tests; it makes no decisions of its own.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any


@dataclass
class Turn:
    text: str = ""
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


class ScriptedModel:
    def __init__(self, turns: list[Turn]) -> None:
        self.turns = list(turns)
        self.requests: list[dict[str, Any]] = []
        self._ids = 0

    @property
    def messages(self) -> ScriptedModel:  # mirrors client.messages.create(...)
        return self

    def create(self, **request: Any) -> SimpleNamespace:
        self.requests.append(copy.deepcopy(request))
        turn = self.turns.pop(0) if self.turns else Turn(text="(end of script)")
        content: list[SimpleNamespace] = []
        if turn.text:
            content.append(SimpleNamespace(type="text", text=turn.text))
        for name, arguments in turn.tool_calls:
            self._ids += 1
            content.append(
                SimpleNamespace(
                    type="tool_use", id=f"toolu_{self._ids:03d}", name=name, input=arguments
                )
            )
        return SimpleNamespace(
            content=content,
            stop_reason="tool_use" if turn.tool_calls else "end_turn",
            usage=SimpleNamespace(input_tokens=0, output_tokens=0),
        )
