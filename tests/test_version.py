from __future__ import annotations

import tomllib
from pathlib import Path

import agentic_research

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_version_matches_pyproject() -> None:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    assert agentic_research.__version__ == project["version"]
