"""Freeze guards for the M1 tool-calling probe (PLAN M1).

The probe file and bar are committed before any probe run. This module only
checks the freeze: exactly 20 tool-expecting prompts and that the on-disk
sha256 matches ``config/probe.toml``. Scoring / CLI live in later commits.
"""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROBE_TOML = ROOT / "config" / "probe.toml"
PROBE_YAML = ROOT / "probes" / "tool_calls.v1.yaml"


def _load_config() -> dict[str, Any]:
    with PROBE_TOML.open("rb") as handle:
        data = tomllib.load(handle)
    assert isinstance(data, dict)
    return data


def _load_probe() -> dict[str, Any]:
    raw = PROBE_YAML.read_bytes()
    data = yaml.safe_load(raw)
    assert isinstance(data, dict)
    return data


def test_probe_toml_has_required_bar_fields() -> None:
    cfg = _load_config()
    assert cfg["probe_file"] == "probes/tool_calls.v1.yaml"
    assert cfg["min_valid"] == 18
    assert cfg["n"] == 20
    assert cfg["rule_version"] == 1
    assert isinstance(cfg["probe_sha256"], str)
    assert len(cfg["probe_sha256"]) == 64
    assert all(c in "0123456789abcdef" for c in cfg["probe_sha256"])
    assert cfg["temperature"] == 0.0
    assert cfg["seed"] == 0
    assert cfg["enable_thinking"] is False
    assert cfg["model"] == "Qwen3-1.7B-Q4_K_M"
    assert cfg["formats"] == ["native", "json"]
    assert cfg["label"] == "smoke-only"


def test_probe_file_sha256_matches_config() -> None:
    cfg = _load_config()
    digest = hashlib.sha256(PROBE_YAML.read_bytes()).hexdigest()
    assert digest == cfg["probe_sha256"]
    assert (ROOT / cfg["probe_file"]).resolve() == PROBE_YAML.resolve()


def test_exactly_twenty_tool_expecting_prompts() -> None:
    cfg = _load_config()
    data = _load_probe()
    prompts = data["prompts"]
    assert isinstance(prompts, list)
    tool_expecting = [p for p in prompts if p.get("expect") == "tool_call"]
    no_tool = [p for p in prompts if p.get("expect") == "no_tool"]
    assert len(tool_expecting) == cfg["n"] == 20
    assert len(no_tool) >= 1
    assert len(prompts) == len(tool_expecting) + len(no_tool)

    ids = [p["id"] for p in prompts]
    assert len(ids) == len(set(ids))

    allowed_tools = set(data["tools"])
    assert allowed_tools == {"calculator", "search_corpus"}
    for prompt in tool_expecting:
        assert prompt["tool"] in allowed_tools
        assert isinstance(prompt["user"], str) and prompt["user"].strip()
    for prompt in no_tool:
        assert "tool" not in prompt
        assert isinstance(prompt["user"], str) and prompt["user"].strip()
