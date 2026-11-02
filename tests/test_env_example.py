"""Checks that `.env.example` documents the configuration and holds no secrets.

`.env.example` is the single list of environment variables (PROJECT_SPEC
section 11). These tests keep it parseable, free of secret values, and in
sync with every `ARP_*` name the spec, plan, ADRs, and README mention.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENV_EXAMPLE = ROOT / ".env.example"
DOCS = [
    ROOT / "PROJECT_SPEC.md",
    ROOT / "PLAN.md",
    ROOT / "README.md",
    *sorted((ROOT / "docs").rglob("*.md")),
]

_ASSIGNMENT = re.compile(r"^(?P<key>[A-Z][A-Z0-9_]*)=(?P<value>.*)$")
_ARP_NAME = re.compile(r"\bARP_[A-Z0-9_]*[A-Z0-9]\b")
_SECRET_HINT = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD)$")

# Values that must be set in the example because the spec fixes them.
SPEC_DEFAULTS = {
    "ARP_PROVIDER": "local",
    "ARP_LOCAL_CTX": "4096",
    "ARP_MAX_CONSECUTIVE_FAILURES": "3",
    "ARP_MAX_RETRIES_PER_RUN": "20",
    "ARP_HOSTED_API_KEY_ENV": "CLINE_API_KEY",
    "ARP_HOSTED_MAX_TOKENS": "1000",
    "ARP_ALLOW_LIVE_FETCH": "false",
}


class EnvExampleError(ValueError):
    """A line in an env file is neither blank, a comment, nor KEY=value."""


def parse_env_example(text: str) -> dict[str, str]:
    """Parse KEY=value lines; reject anything else and duplicate keys."""
    entries: dict[str, str] = {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _ASSIGNMENT.match(line)
        if match is None:
            raise EnvExampleError(f"line {lineno}: not KEY=value: {line!r}")
        key, value = match["key"], match["value"]
        if key in entries:
            raise EnvExampleError(f"line {lineno}: duplicate key {key}")
        if value != value.strip() or value.startswith(("'", '"')):
            raise EnvExampleError(f"line {lineno}: {key} has quotes or padding")
        entries[key] = value
    return entries


@pytest.fixture(scope="module")
def env() -> dict[str, str]:
    return parse_env_example(ENV_EXAMPLE.read_text(encoding="utf-8"))


def test_every_documented_arp_variable_is_listed(env: dict[str, str]) -> None:
    documented = {name for doc in DOCS for name in _ARP_NAME.findall(doc.read_text("utf-8"))}
    assert documented, "expected the docs to mention ARP_* variables"
    assert sorted(documented - env.keys()) == []


def test_no_undocumented_arp_variables(env: dict[str, str]) -> None:
    documented = {name for doc in DOCS for name in _ARP_NAME.findall(doc.read_text("utf-8"))}
    listed = {key for key in env if key.startswith("ARP_")}
    assert sorted(listed - documented) == []


def test_secret_values_are_empty(env: dict[str, str]) -> None:
    secrets = {key: value for key, value in env.items() if _SECRET_HINT.search(key)}
    assert "CLINE_API_KEY" in secrets
    assert {key: value for key, value in secrets.items() if value} == {}


def test_hosted_key_env_names_a_listed_empty_variable(env: dict[str, str]) -> None:
    key_var = env["ARP_HOSTED_API_KEY_ENV"]
    assert key_var in env
    assert env[key_var] == ""


@pytest.mark.parametrize(("key", "value"), sorted(SPEC_DEFAULTS.items()))
def test_spec_defaults(env: dict[str, str], key: str, value: str) -> None:
    assert env[key] == value


def test_tool_format_unset_until_probe(env: dict[str, str]) -> None:
    # The default comes from the M1 probe's pre-registered rule, not a guess.
    assert env["ARP_TOOL_FORMAT"] in {"", "native", "json"}


def test_local_base_url_is_loopback(env: dict[str, str]) -> None:
    assert re.fullmatch(r"http://127\.0\.0\.1:\d+/v1", env["ARP_LLM_BASE_URL"])


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("export FOO=1\n", "not KEY=value"),
        ("lower=1\n", "not KEY=value"),
        ("FOO=1\nFOO=2\n", "duplicate key FOO"),
        ('FOO="1"\n', "quotes or padding"),
        ("FOO= 1\n", "quotes or padding"),
    ],
)
def test_parser_rejects_malformed_lines(text: str, message: str) -> None:
    with pytest.raises(EnvExampleError, match=message):
        parse_env_example(text)


def test_parser_skips_comments_and_blank_lines() -> None:
    assert parse_env_example("# c\n\n  # indented\nA=1\nB=\n") == {"A": "1", "B": ""}
