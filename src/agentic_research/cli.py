"""Command-line entry point (`arp`).

Implemented: ``arp tools schema``. Later milestones add ``doctor``,
``probe tool-calls``, ``run``, ``eval``, and ``trace show``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from agentic_research import __version__
from agentic_research.tools.builtin import default_registry


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
