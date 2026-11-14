"""Command-line entry point (`arp`).

Implemented: ``arp tools schema``, ``arp doctor``, ``arp probe tool-calls``,
``arp run --provider scripted --script FILE`` and ``arp trace show FILE``.
Later milestones add ``eval`` and model-backed ``run`` providers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from agentic_research import __version__
from agentic_research.agent import AgentLoop, Budgets, Task
from agentic_research.agent.state import RunStatus
from agentic_research.policies import LLMPolicy
from agentic_research.preflight import run_doctor
from agentic_research.probe import run_probe_cli
from agentic_research.providers.scripted import SCRIPTED_MODEL, ScriptedProvider, load_script
from agentic_research.tools.builtin import default_registry
from agentic_research.tools.executor import ToolExecutor
from agentic_research.trace import TraceError, read_trace, write_trace
from agentic_research.trace.show import format_trace

#: Where ``arp run`` writes traces when ``--trace-out`` is not given.
DEFAULT_TRACE_DIR = Path("traces")


def _tools_schema(args: argparse.Namespace) -> int:
    registry = default_registry()
    schemas = registry.openai_schemas()
    if args.tool is not None:
        if args.tool not in registry:
            known = ", ".join(registry.names)
            print(f"arp: unknown tool {args.tool!r} (known: {known})", file=sys.stderr)
            return 2
        schemas = [s for s in schemas if s["function"]["name"] == args.tool]
    print(json.dumps(schemas, indent=2, ensure_ascii=False))
    return 0


def _doctor(_args: argparse.Namespace) -> int:
    report = run_doctor()
    print(report.format())
    return 0 if report.ok else 1


def _probe_tool_calls(_args: argparse.Namespace) -> int:
    return run_probe_cli()


def _fail(message: str) -> int:
    print(f"arp: {message}", file=sys.stderr)
    return 2


def _default_run_id() -> str:
    return datetime.now(UTC).strftime("run-%Y%m%dT%H%M%SZ")


def _run(args: argparse.Namespace) -> int:
    """``arp run``: one agent run; exit 0 if it answered, 1 if not, 2 on bad input."""
    try:
        script = load_script(args.script)
    except OSError as exc:
        return _fail(f"cannot read script {args.script}: {exc.strerror or exc}")
    except ValidationError as exc:
        first = exc.errors(include_url=False, include_input=False)[0]
        where = ".".join(str(part) for part in first["loc"]) or "script"
        return _fail(f"invalid script {args.script}: {where}: {first['msg']}")
    question = args.question if args.question is not None else script.question
    if not question:
        return _fail("no question: the script has none, so pass --question")

    run_id = args.run_id if args.run_id is not None else _default_run_id()
    registry = default_registry()
    provider = ScriptedProvider.from_script(script)
    policy = LLMPolicy(provider, registry, tool_format=args.tool_format, model=SCRIPTED_MODEL)
    budgets = Budgets(
        max_steps=args.max_steps,
        max_tool_calls=args.max_tool_calls,
        max_tokens=args.max_tokens,
        deadline_s=args.deadline_s,
    )
    try:
        loop = AgentLoop(policy, ToolExecutor(registry, seed=args.seed), budgets, run_id=run_id)
    except ValueError as exc:
        return _fail(str(exc))
    trace_out: Path = args.trace_out or DEFAULT_TRACE_DIR / f"{run_id}.jsonl"
    if trace_out.exists():
        return _fail(f"{trace_out} already exists; pick another --trace-out or --run-id")

    traced = asyncio.run(
        loop.run_traced(
            Task(question=question),
            provider=provider.name,
            model=SCRIPTED_MODEL,
            tool_format=args.tool_format,
        )
    )
    trace_out.parent.mkdir(parents=True, exist_ok=True)
    write_trace(traced.trace, trace_out)
    state = traced.state
    print(f"run {run_id}: {state.terminal_label}")
    if state.answer is not None:
        print(f"answer: {state.answer}")
    print(f"trace: {trace_out}")
    return 0 if state.status is RunStatus.SUCCEEDED else 1


def _trace_show(args: argparse.Namespace) -> int:
    try:
        trace = read_trace(args.file)
    except OSError as exc:
        return _fail(f"cannot read trace {args.file}: {exc.strerror or exc}")
    except TraceError as exc:
        return _fail(str(exc))
    print(format_trace(trace, max_chars=args.max_chars))
    return 0


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def _positive_float(text: str) -> float:
    value = float(text)
    if not value > 0 or value == float("inf"):
        raise argparse.ArgumentTypeError("must be a finite number > 0")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arp", description="RabbitHole research agent (agentic-research-platform)"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    tools = commands.add_parser("tools", help="inspect the built-in tools")
    tools_commands = tools.add_subparsers(dest="tools_command", metavar="COMMAND")
    schema = tools_commands.add_parser(
        "schema", help="print tool definitions in the OpenAI chat-completions `tools` format"
    )
    schema.add_argument("--tool", metavar="NAME", help="print only this tool")
    schema.set_defaults(handler=_tools_schema)
    tools.set_defaults(help_parser=tools)

    doctor = commands.add_parser(
        "doctor",
        help="memory preflight and local-server check before live runs",
    )
    doctor.set_defaults(handler=_doctor)

    probe = commands.add_parser("probe", help="frozen evaluation probes")
    probe_commands = probe.add_subparsers(dest="probe_command", metavar="COMMAND")
    tool_calls = probe_commands.add_parser(
        "tool-calls",
        help="run the frozen native-vs-json tool-calling probe (local model; slow)",
    )
    tool_calls.set_defaults(handler=_probe_tool_calls)
    probe.set_defaults(help_parser=probe)

    run = commands.add_parser(
        "run",
        help="run the agent on one question and write its trace",
        description=(
            "Run the agent loop once and write a JSONL trace. Only the scripted "
            "provider exists so far: it replays a script file's canned replies "
            "(no model). Exit status: 0 answered, 1 any other end, 2 bad input."
        ),
    )
    run.add_argument("--provider", required=True, choices=["scripted"])
    run.add_argument("--script", required=True, type=Path, metavar="FILE")
    run.add_argument("--question", help="the task (default: the script's question)")
    run.add_argument("--tool-format", choices=["native", "json"], default="native")
    run.add_argument("--max-steps", type=_positive_int, default=8, metavar="N")
    run.add_argument("--max-tool-calls", type=_positive_int, metavar="N")
    run.add_argument("--max-tokens", type=_positive_int, metavar="N")
    run.add_argument("--deadline-s", type=_positive_float, metavar="SECONDS")
    run.add_argument("--seed", type=int, default=0, help="retry-jitter seed (default 0)")
    run.add_argument("--run-id", help="trace run id (default: run-<UTC timestamp>)")
    run.add_argument(
        "--trace-out",
        type=Path,
        metavar="PATH",
        help=f"trace file to create (default: {DEFAULT_TRACE_DIR}/<run-id>.jsonl)",
    )
    run.set_defaults(handler=_run)

    trace = commands.add_parser("trace", help="inspect run traces")
    trace_commands = trace.add_subparsers(dest="trace_command", metavar="COMMAND")
    show = trace_commands.add_parser("show", help="print a trace file as a span tree")
    show.add_argument("file", type=Path, metavar="FILE")
    show.add_argument(
        "--max-chars", type=_positive_int, default=200, help="shorten values longer than this"
    )
    show.set_defaults(handler=_trace_show)
    trace.set_defaults(help_parser=trace)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        getattr(args, "help_parser", parser).print_help()
        return 0
    result: int = handler(args)
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
