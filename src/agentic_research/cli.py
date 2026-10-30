"""Command-line entry point (`arp`).

Subcommands (`tools schema`, `doctor`, `probe tool-calls`, `run`, `eval`) are
added by later milestones; for now the CLI only reports its version.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from agentic_research import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arp", description="RabbitHole research agent (agentic-research-platform)"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
