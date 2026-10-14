"""`guarded-agent run | demo | trace`."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from guarded_agent.agent import Agent
from guarded_agent.builtin_tools import calculator, convert_units, email_tool, notes_tools
from guarded_agent.demo import run_demo
from guarded_agent.guardrails import Policy, approve_all, deny_all
from guarded_agent.tools import ToolRegistry
from guarded_agent.tracing import Trace, load_trace, render


def ask_user(name: str, arguments: dict[str, Any]) -> bool:
    reply = input(f"\nAllow {name}({json.dumps(arguments)})? [y/N] ")
    return reply.strip().lower() in {"y", "yes"}


def cmd_run(args: argparse.Namespace) -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run with Claude, or try `guarded-agent demo`.")
        return 2
    domains = set(os.environ.get("GUARDED_AGENT_EMAIL_DOMAINS", "example.com").split(","))
    registry = ToolRegistry().add(
        calculator, convert_units, email_tool(args.outbox, allowed_domains=domains)
    )
    if args.notes:
        registry.add(*notes_tools(args.notes))
    approver = {"ask": ask_user, "yes": approve_all, "no": deny_all}[args.approve]
    agent = Agent(registry, Policy(approver=approver, max_steps=args.max_steps))
    trace = Trace(args.traces)
    result = agent.run(args.task, trace)
    print(result.answer)
    print(f"\n[{result.stopped}; {result.tool_calls} tool calls; trace: {trace.path}]")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory() as workdir:
        result, trace = run_demo(Path(workdir), args.traces)
    print(render(trace.events))
    print(f"\n{result.tool_calls} tool calls, {result.blocked} blocked or denied")
    if trace.path:
        print(f"Trace written to {trace.path}")
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="guarded-agent", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Run a task with Claude")
    run.add_argument("task")
    run.add_argument("--notes", type=Path, help="Folder of .md/.txt notes the agent may read")
    run.add_argument("--outbox", type=Path, default=Path("outbox"))
    run.add_argument("--approve", choices=["ask", "yes", "no"], default="ask")
    run.add_argument("--max-steps", type=int, default=8)
    run.add_argument("--traces", type=Path, default=Path("traces"))

    demo = commands.add_parser("demo", help="Offline scripted run showing the guardrails")
    demo.add_argument("--traces", type=Path, default=None)

    show = commands.add_parser("trace", help="Print a saved trace")
    show.add_argument("path", type=Path)

    args = parser.parse_args(argv)
    if args.command == "run":
        sys.exit(cmd_run(args))
    if args.command == "demo":
        sys.exit(cmd_demo(args))
    print(render(load_trace(args.path)))


if __name__ == "__main__":
    main()
