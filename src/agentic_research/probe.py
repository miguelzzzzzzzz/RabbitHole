"""M1 tool-calling probe: native vs JSON-in-text on a frozen prompt set.

``arp probe tool-calls`` loads ``config/probe.toml`` + ``probes/tool_calls.v1.yaml``,
refuses a dirty tree or a probe-file sha256 mismatch, runs both formats against
a local ``OpenAICompatibleProvider``, scores replies, applies the pre-registered
decision rule, and writes a ``smoke-only`` report under ``evals/results/``.

The live path is marked ``slow`` and never runs in CI. Scoring helpers, Wilson
intervals, the decision rule, and the freeze guards are unit-tested offline.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import subprocess
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, Protocol

import yaml
from pydantic import Field

from agentic_research.preflight import DoctorReport, run_doctor
from agentic_research.providers.openai_compat import OpenAICompatibleProvider
from agentic_research.providers.toolcall_json import (
    ParsedFinalAnswer,
    ParsedToolCall,
    ParseFailureCounter,
    parse_toolcall_json,
)
from agentic_research.providers.types import ChatMessage, ChatRequest, ChatResponse
from agentic_research.tools.base import RetryPolicy, ToolArgs, ToolContext, ToolError, ToolOutput
from agentic_research.tools.calculator import CalculatorTool
from agentic_research.tools.registry import ToolRegistry

ExpectKind = Literal["tool_call", "no_tool"]
FormatName = Literal["native", "json"]

#: Repo-relative paths pinned by the freeze commit.
DEFAULT_CONFIG_PATH: Final = Path("config/probe.toml")
DEFAULT_PROBE_PATH: Final = Path("probes/tool_calls.v1.yaml")

JSON_PROTOCOL_INSTRUCTIONS: Final = (
    "You must reply with exactly one JSON object, optionally inside a single "
    "```json fence. Shape:\n"
    '{"action":"tool_call","tool":"<name>","arguments":{...}}\n'
    "or\n"
    '{"action":"final_answer","answer":"..."}\n'
    "No other keys, no prose outside the object/fence, no guessing."
)

NATIVE_SYSTEM: Final = (
    "You are a research assistant with tools. Use a tool when the user asks "
    "you to compute something or search the corpus. For simple chitchat or "
    "general-knowledge questions that do not need a tool, answer directly "
    "without calling a tool."
)


class ProbeError(RuntimeError):
    """Base error for probe configuration / freeze problems."""


class ProbeRefuseError(ProbeError):
    """The probe refused to run (dirty tree, hash mismatch, doctor fail)."""


class ChatCompleter(Protocol):
    """Minimal async chat surface used by the probe (real provider or mock)."""

    async def complete(self, request: ChatRequest) -> ChatResponse: ...


# --- Stub search_corpus (schema only; M3 wires the real tool) -----------------


class SearchCorpusArgs(ToolArgs):
    """Stub argument model matching the planned M3 ``search_corpus`` shape."""

    query: str = Field(min_length=1, description="Natural-language search query.")
    k: int = Field(default=5, ge=1, le=50, description="Number of passages to return.")
    filters: dict[str, str] | None = Field(
        default=None, description="Optional metadata filters (string keys and values)."
    )


class SearchCorpusOutput(ToolOutput):
    hits: list[dict[str, Any]] = Field(default_factory=list)


class SearchCorpusStubTool:
    """Schema-only stub so the probe can export and validate ``search_corpus``."""

    name = "search_corpus"
    description = (
        "Search the research corpus for relevant passages. Use this when the "
        "user asks to find, look up, or search documents or passages."
    )
    args_model: type[ToolArgs] = SearchCorpusArgs
    output_model: type[ToolOutput] = SearchCorpusOutput
    timeout_s = 30.0
    retry = RetryPolicy.no_retry()

    async def run(self, args: SearchCorpusArgs, ctx: ToolContext) -> SearchCorpusOutput:
        raise NotImplementedError("search_corpus is a probe stub; wired in M3")


def probe_registry() -> ToolRegistry:
    """Registry used by the probe: real ``calculator`` + stub ``search_corpus``."""
    return ToolRegistry([CalculatorTool(), SearchCorpusStubTool()])


def args_models(registry: ToolRegistry | None = None) -> dict[str, type[ToolArgs]]:
    reg = registry if registry is not None else probe_registry()
    return {name: reg.get(name).args_model for name in reg.names}


# --- Config / probe file loading ---------------------------------------------


@dataclass(frozen=True)
class ProbeConfig:
    probe_file: str
    probe_sha256: str
    min_valid: int
    n: int
    rule_version: int
    temperature: float
    seed: int
    enable_thinking: bool
    model: str
    formats: tuple[FormatName, ...]
    label: str

    @property
    def probe_path(self) -> Path:
        return Path(self.probe_file)


@dataclass(frozen=True)
class ProbePrompt:
    id: str
    expect: ExpectKind
    user: str
    tool: str | None = None


@dataclass(frozen=True)
class ProbeSet:
    version: int
    id: str
    tools: tuple[str, ...]
    prompts: tuple[ProbePrompt, ...]

    @property
    def tool_expecting(self) -> tuple[ProbePrompt, ...]:
        return tuple(p for p in self.prompts if p.expect == "tool_call")

    @property
    def no_tool(self) -> tuple[ProbePrompt, ...]:
        return tuple(p for p in self.prompts if p.expect == "no_tool")


def find_repo_root(start: Path | None = None) -> Path:
    """Walk parents until a ``.git`` directory is found."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists():
            return candidate
    raise ProbeError(f"cannot find git repo root from {here}")


def load_probe_config(repo_root: Path, config_path: Path | None = None) -> ProbeConfig:
    path = repo_root / (config_path or DEFAULT_CONFIG_PATH)
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    formats_raw = raw["formats"]
    formats: list[FormatName] = []
    for item in formats_raw:
        if item not in ("native", "json"):
            raise ProbeError(f"unknown format in probe.toml: {item!r}")
        formats.append(item)
    return ProbeConfig(
        probe_file=str(raw["probe_file"]),
        probe_sha256=str(raw["probe_sha256"]),
        min_valid=int(raw["min_valid"]),
        n=int(raw["n"]),
        rule_version=int(raw["rule_version"]),
        temperature=float(raw["temperature"]),
        seed=int(raw["seed"]),
        enable_thinking=bool(raw["enable_thinking"]),
        model=str(raw["model"]),
        formats=tuple(formats),
        label=str(raw["label"]),
    )


def load_probe_set(path: Path) -> ProbeSet:
    data = yaml.safe_load(path.read_bytes())
    if not isinstance(data, dict):
        raise ProbeError(f"probe file must be a mapping: {path}")
    prompts: list[ProbePrompt] = []
    for entry in data["prompts"]:
        expect = entry["expect"]
        if expect not in ("tool_call", "no_tool"):
            raise ProbeError(f"unknown expect {expect!r} in prompt {entry.get('id')!r}")
        prompts.append(
            ProbePrompt(
                id=str(entry["id"]),
                expect=expect,
                user=str(entry["user"]),
                tool=str(entry["tool"]) if "tool" in entry else None,
            )
        )
    return ProbeSet(
        version=int(data["version"]),
        id=str(data["id"]),
        tools=tuple(str(t) for t in data["tools"]),
        prompts=tuple(prompts),
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_probe_hash(cfg: ProbeConfig, probe_path: Path) -> str:
    """Return the digest, or raise ``ProbeRefuseError`` on mismatch."""
    digest = file_sha256(probe_path)
    if digest != cfg.probe_sha256:
        raise ProbeRefuseError(
            f"probe file sha256 mismatch: got {digest}, "
            f"config expects {cfg.probe_sha256}. Refusing to run "
            "(no request sent). Bump to a new probe version instead of editing."
        )
    return digest


def git_working_tree_dirty(repo_root: Path) -> bool:
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(proc.stdout.strip())


def assert_clean_tree(repo_root: Path) -> None:
    if git_working_tree_dirty(repo_root):
        raise ProbeRefuseError(
            "git working tree is dirty; refuse to run the probe "
            "(no request sent). Commit or stash changes first."
        )


def freeze_commit_sha(repo_root: Path, probe_rel: str = "probes/tool_calls.v1.yaml") -> str:
    """SHA of the commit that first added the frozen probe file."""
    proc = subprocess.run(
        ["git", "log", "--diff-filter=A", "--format=%H", "--", probe_rel],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise ProbeError(f"cannot find freeze commit for {probe_rel}")
    # git log lists newest first; the adding commit is the last for a single-add file.
    return lines[-1]


def head_commit_sha(repo_root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


# --- Wilson interval + decision rule -----------------------------------------


def wilson_interval(k: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    """Two-sided Wilson score interval for ``k`` successes in ``n`` trials."""
    if n <= 0:
        raise ValueError(f"n must be > 0 (got {n})")
    if k < 0 or k > n:
        raise ValueError(f"k must be in [0, n] (got k={k}, n={n})")
    if z <= 0:
        raise ValueError(f"z must be > 0 (got {z})")
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)) / denom
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return lo, hi


def decide_default(
    *,
    native_schema_valid: int,
    json_schema_valid: int,
    min_valid: int,
    n: int,
) -> FormatName:
    """Pre-registered rule: native only if >= min_valid and not below json."""
    if native_schema_valid < 0 or json_schema_valid < 0:
        raise ValueError("counts must be non-negative")
    if native_schema_valid > n or json_schema_valid > n:
        raise ValueError(f"counts must be <= n ({n})")
    if native_schema_valid >= min_valid and native_schema_valid >= json_schema_valid:
        return "native"
    return "json"


# --- Scoring -----------------------------------------------------------------


@dataclass(frozen=True)
class PromptScore:
    prompt_id: str
    expect: ExpectKind
    expected_tool: str | None
    structurally_valid: bool
    correct_tool_name: bool
    schema_valid_args: bool
    correct_no_tool: bool
    schema_valid: bool
    parse_failure_reason: str | None
    latency_ms: float
    tool_name: str | None
    arguments: dict[str, Any] | None
    raw_content: str | None
    raw_tool_calls: list[dict[str, Any]] | None


def _tool_calls_as_dicts(response: ChatResponse) -> list[dict[str, Any]] | None:
    if response.message.tool_calls is None:
        return None
    out: list[dict[str, Any]] = []
    for call in response.message.tool_calls:
        out.append(
            {
                "id": call.id,
                "type": call.type,
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                    "arguments_raw": call.function.arguments_raw,
                    "parse_error": call.function.parse_error,
                },
            }
        )
    return out


def score_native_reply(
    prompt: ProbePrompt,
    response: ChatResponse,
    registry: ToolRegistry,
) -> PromptScore:
    """Score one native ``tools`` / ``tool_calls`` reply against a prompt."""
    latency = response.latency_ms
    content = response.message.content
    raw_calls = _tool_calls_as_dicts(response)
    calls = response.message.tool_calls

    if prompt.expect == "no_tool":
        correct = calls is None or len(calls) == 0
        return PromptScore(
            prompt_id=prompt.id,
            expect=prompt.expect,
            expected_tool=None,
            structurally_valid=False,
            correct_tool_name=False,
            schema_valid_args=False,
            correct_no_tool=correct,
            schema_valid=False,
            parse_failure_reason=None if correct else "unexpected_tool_call",
            latency_ms=latency,
            tool_name=None,
            arguments=None,
            raw_content=content,
            raw_tool_calls=raw_calls,
        )

    expected = prompt.tool
    if calls is None or len(calls) == 0:
        return PromptScore(
            prompt_id=prompt.id,
            expect=prompt.expect,
            expected_tool=expected,
            structurally_valid=False,
            correct_tool_name=False,
            schema_valid_args=False,
            correct_no_tool=False,
            schema_valid=False,
            parse_failure_reason="missing_tool_call",
            latency_ms=latency,
            tool_name=None,
            arguments=None,
            raw_content=content,
            raw_tool_calls=raw_calls,
        )

    # Score the first call (probe prompts expect a single tool use).
    first = calls[0]
    name = first.function.name
    structurally_valid = True
    correct_name = name == expected
    args = first.function.arguments
    args_ok = False
    parse_reason = first.function.parse_error
    if first.function.arguments_valid and args is not None and expected is not None:
        validated = registry.validate_args(expected if correct_name else name, args)
        args_ok = not isinstance(validated, ToolError) and correct_name
        if isinstance(validated, ToolError):
            parse_reason = validated.reason
    elif parse_reason is None and args is None:
        parse_reason = "missing_arguments"

    schema_valid = structurally_valid and correct_name and args_ok
    return PromptScore(
        prompt_id=prompt.id,
        expect=prompt.expect,
        expected_tool=expected,
        structurally_valid=structurally_valid,
        correct_tool_name=correct_name,
        schema_valid_args=args_ok,
        correct_no_tool=False,
        schema_valid=schema_valid,
        parse_failure_reason=None if schema_valid else (parse_reason or "invalid_call"),
        latency_ms=latency,
        tool_name=name,
        arguments=args,
        raw_content=content,
        raw_tool_calls=raw_calls,
    )


def score_json_reply(
    prompt: ProbePrompt,
    response: ChatResponse,
    tool_models: Mapping[str, type[ToolArgs]],
) -> PromptScore:
    """Score one JSON-in-text reply against a prompt."""
    latency = response.latency_ms
    content = response.message.content or ""
    failures = ParseFailureCounter()
    result = parse_toolcall_json(content, tool_models, failures=failures)

    if prompt.expect == "no_tool":
        correct = result.ok and isinstance(result.action, ParsedFinalAnswer)
        reason = (
            None
            if correct
            else (result.failure.reason if result.failure is not None else "expected_final_answer")
        )
        return PromptScore(
            prompt_id=prompt.id,
            expect=prompt.expect,
            expected_tool=None,
            structurally_valid=False,
            correct_tool_name=False,
            schema_valid_args=False,
            correct_no_tool=correct,
            schema_valid=False,
            parse_failure_reason=reason,
            latency_ms=latency,
            tool_name=None,
            arguments=None,
            raw_content=content,
            raw_tool_calls=None,
        )

    expected = prompt.tool
    if not result.ok or not isinstance(result.action, ParsedToolCall):
        reason = result.failure.reason if result.failure is not None else "not_a_tool_call"
        return PromptScore(
            prompt_id=prompt.id,
            expect=prompt.expect,
            expected_tool=expected,
            structurally_valid=False,
            correct_tool_name=False,
            schema_valid_args=False,
            correct_no_tool=False,
            schema_valid=False,
            parse_failure_reason=reason,
            latency_ms=latency,
            tool_name=None,
            arguments=None,
            raw_content=content,
            raw_tool_calls=None,
        )

    action = result.action
    # parse_toolcall_json already schema-validated arguments against the allowlist.
    structurally_valid = True
    correct_name = action.tool == expected
    args_ok = correct_name  # invalid args would have failed the parse
    schema_valid = structurally_valid and correct_name and args_ok
    return PromptScore(
        prompt_id=prompt.id,
        expect=prompt.expect,
        expected_tool=expected,
        structurally_valid=structurally_valid,
        correct_tool_name=correct_name,
        schema_valid_args=args_ok,
        correct_no_tool=False,
        schema_valid=schema_valid,
        parse_failure_reason=None if schema_valid else "wrong_tool",
        latency_ms=latency,
        tool_name=action.tool,
        arguments=dict(action.arguments),
        raw_content=content,
        raw_tool_calls=None,
    )


@dataclass(frozen=True)
class FormatResult:
    format: FormatName
    scores: tuple[PromptScore, ...]
    schema_valid_count: int
    n_tool_expecting: int
    no_tool_correct: int
    n_no_tool: int
    parse_failures: int
    wilson_lo: float
    wilson_hi: float

    @property
    def raw_count_label(self) -> str:
        return f"{self.schema_valid_count}/{self.n_tool_expecting}"


def summarize_format(
    format_name: FormatName,
    scores: Sequence[PromptScore],
    *,
    n_tool_expecting: int,
) -> FormatResult:
    tool_scores = [s for s in scores if s.expect == "tool_call"]
    no_tool_scores = [s for s in scores if s.expect == "no_tool"]
    schema_valid = sum(1 for s in tool_scores if s.schema_valid)
    no_tool_correct = sum(1 for s in no_tool_scores if s.correct_no_tool)
    parse_failures = sum(1 for s in scores if s.parse_failure_reason is not None)
    lo, hi = wilson_interval(schema_valid, n_tool_expecting)
    return FormatResult(
        format=format_name,
        scores=tuple(scores),
        schema_valid_count=schema_valid,
        n_tool_expecting=n_tool_expecting,
        no_tool_correct=no_tool_correct,
        n_no_tool=len(no_tool_scores),
        parse_failures=parse_failures,
        wilson_lo=lo,
        wilson_hi=hi,
    )


# --- Running ------------------------------------------------------------------


def _system_for_format(format_name: FormatName, registry: ToolRegistry) -> str:
    if format_name == "native":
        return NATIVE_SYSTEM
    schema_lines = []
    for schema in registry.openai_schemas():
        fn = schema["function"]
        schema_lines.append(
            f"- {fn['name']}: {fn['description']} parameters={json.dumps(fn['parameters'])}"
        )
    return (
        JSON_PROTOCOL_INSTRUCTIONS
        + "\nAvailable tools:\n"
        + "\n".join(schema_lines)
        + "\n"
        + NATIVE_SYSTEM
    )


def build_request(
    prompt: ProbePrompt,
    *,
    format_name: FormatName,
    registry: ToolRegistry,
    cfg: ProbeConfig,
) -> ChatRequest:
    system = _system_for_format(format_name, registry)
    messages = (
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=prompt.user),
    )
    tools = tuple(registry.openai_schemas()) if format_name == "native" else None
    return ChatRequest(
        messages=messages,
        tools=tools,
        model=cfg.model,
        temperature=cfg.temperature,
        seed=cfg.seed,
        enable_thinking=cfg.enable_thinking,
    )


async def run_format(
    format_name: FormatName,
    probe: ProbeSet,
    *,
    provider: ChatCompleter,
    registry: ToolRegistry,
    cfg: ProbeConfig,
) -> FormatResult:
    tool_models = args_models(registry)
    scores: list[PromptScore] = []
    for prompt in probe.prompts:
        request = build_request(prompt, format_name=format_name, registry=registry, cfg=cfg)
        response = await provider.complete(request)
        if format_name == "native":
            scores.append(score_native_reply(prompt, response, registry))
        else:
            scores.append(score_json_reply(prompt, response, tool_models))
    return summarize_format(format_name, scores, n_tool_expecting=cfg.n)


@dataclass(frozen=True)
class ProbeRunResult:
    config: ProbeConfig
    probe_sha256: str
    freeze_commit_sha: str
    head_commit_sha: str
    chosen_default: FormatName
    formats: dict[FormatName, FormatResult]
    doctor: dict[str, Any]
    metadata: dict[str, Any]
    label: str

    def to_report(self) -> dict[str, Any]:
        formats_out: dict[str, Any] = {}
        for name, fr in self.formats.items():
            formats_out[name] = {
                "schema_valid": fr.schema_valid_count,
                "n": fr.n_tool_expecting,
                "raw": fr.raw_count_label,
                "wilson_95": [round(fr.wilson_lo, 6), round(fr.wilson_hi, 6)],
                "no_tool_correct": fr.no_tool_correct,
                "n_no_tool": fr.n_no_tool,
                "parse_failures": fr.parse_failures,
                "scores": [
                    {
                        "prompt_id": s.prompt_id,
                        "expect": s.expect,
                        "expected_tool": s.expected_tool,
                        "structurally_valid": s.structurally_valid,
                        "correct_tool_name": s.correct_tool_name,
                        "schema_valid_args": s.schema_valid_args,
                        "correct_no_tool": s.correct_no_tool,
                        "schema_valid": s.schema_valid,
                        "parse_failure_reason": s.parse_failure_reason,
                        "latency_ms": s.latency_ms,
                        "tool_name": s.tool_name,
                        "arguments": s.arguments,
                        "raw_content": s.raw_content,
                        "raw_tool_calls": s.raw_tool_calls,
                    }
                    for s in fr.scores
                ],
            }
        return {
            "label": self.label,
            "headline_eligible": False,
            "rule_version": self.config.rule_version,
            "decision_rule": (
                "native becomes default only if schema-valid on at least "
                f"{self.config.min_valid}/{self.config.n} tool-expecting prompts "
                "and its count is not lower than json's; otherwise default is json"
            ),
            "chosen_default": self.chosen_default,
            "probe_file": self.config.probe_file,
            "probe_sha256": self.probe_sha256,
            "freeze_commit_sha": self.freeze_commit_sha,
            "head_commit_sha": self.head_commit_sha,
            "min_valid": self.config.min_valid,
            "n": self.config.n,
            "temperature": self.config.temperature,
            "seed": self.config.seed,
            "enable_thinking": self.config.enable_thinking,
            "model": self.config.model,
            "formats": formats_out,
            "doctor": self.doctor,
            "metadata": self.metadata,
            "note": (
                "n=20 gives a wide Wilson interval; passing the bar shows "
                "'probably usable', not '90% reliable'. Reliability is measured "
                "by parse_failures on full M5 runs."
            ),
        }


def _doctor_as_dict(report: DoctorReport) -> dict[str, Any]:
    return {
        "ok": report.ok,
        "formatted": report.format(),
        "required_mib": report.required_mib,
        "mem_available_mib": report.mem_available_mib,
        "checks": [
            {
                "name": c.name,
                "ok": c.ok,
                "reason": c.reason,
                "message": c.message,
            }
            for c in report.checks
        ],
    }


def runtime_metadata(repo_root: Path) -> dict[str, Any]:
    """Pins and date for the report (from docs / local runtime, not scraped)."""
    return {
        "date": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "model_gguf_sha256": "b139949c5bd74937ad8ed8c8cf3d9ffb1e99c866c823204dc42c0d91fa181897",
        "llama_cpp_build": "b12032",
        "llama_cpp_version": "0.8.0-dev (build 12032, commit dde879bc4)",
        "launch_flags": (
            "llama-server -m <GGUF> --alias Qwen3-1.7B-Q4_K_M "
            "--host 127.0.0.1 --port 8080 -c 4096 -np 1 --jinja "
            "--cache-ram 0 --seed 0"
        ),
        "source": str(repo_root / "docs" / "local-models.md"),
    }


async def run_probe(
    *,
    repo_root: Path | None = None,
    provider: ChatCompleter | None = None,
    skip_doctor: bool = False,
    skip_clean_tree: bool = False,
    config_path: Path | None = None,
) -> ProbeRunResult:
    """Load freeze, guard, run both formats, decide default, return result.

    ``skip_doctor`` / ``skip_clean_tree`` exist for unit tests that inject a
    mock provider. The CLI never sets them.
    """
    root = repo_root or find_repo_root()
    cfg = load_probe_config(root, config_path)
    probe_path = root / cfg.probe_path
    if not skip_clean_tree:
        assert_clean_tree(root)
    digest = assert_probe_hash(cfg, probe_path)
    probe = load_probe_set(probe_path)
    if len(probe.tool_expecting) != cfg.n:
        raise ProbeError(
            f"probe has {len(probe.tool_expecting)} tool-expecting prompts; config n={cfg.n}"
        )

    doctor_payload: dict[str, Any]
    if skip_doctor:
        doctor_payload = {"ok": True, "skipped": True, "checks": []}
    else:
        report = run_doctor()
        if not report.ok:
            raise ProbeRefuseError(f"arp doctor failed; refusing probe:\n{report.format()}")
        doctor_payload = _doctor_as_dict(report)

    registry = probe_registry()
    owns_provider = provider is None
    client: ChatCompleter
    if provider is None:
        client = OpenAICompatibleProvider(
            model=cfg.model,
            temperature=cfg.temperature,
            seed=cfg.seed,
            enable_thinking=cfg.enable_thinking,
        )
    else:
        client = provider

    formats: dict[FormatName, FormatResult] = {}
    try:
        for format_name in cfg.formats:
            formats[format_name] = await run_format(
                format_name,
                probe,
                provider=client,
                registry=registry,
                cfg=cfg,
            )
    finally:
        if owns_provider and isinstance(client, OpenAICompatibleProvider):
            await client.aclose()

    native_count = formats["native"].schema_valid_count if "native" in formats else 0
    json_count = formats["json"].schema_valid_count if "json" in formats else 0
    chosen = decide_default(
        native_schema_valid=native_count,
        json_schema_valid=json_count,
        min_valid=cfg.min_valid,
        n=cfg.n,
    )
    return ProbeRunResult(
        config=cfg,
        probe_sha256=digest,
        freeze_commit_sha=freeze_commit_sha(root),
        head_commit_sha=head_commit_sha(root),
        chosen_default=chosen,
        formats=formats,
        doctor=doctor_payload,
        metadata=runtime_metadata(root),
        label=cfg.label,
    )


def write_report(result: ProbeRunResult, repo_root: Path, path: Path | None = None) -> Path:
    """Write the smoke-only JSON report; return the path used."""
    if path is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = repo_root / "evals" / "results" / f"m1-toolcall-probe-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_report(), indent=2, ensure_ascii=False) + "\n")
    return path


def format_providers_md(result: ProbeRunResult) -> str:
    """Render ``docs/providers.md`` from a probe result."""
    native = result.formats["native"]
    json_fmt = result.formats["json"]
    meta = result.metadata
    lines = [
        "# Providers",
        "",
        "Default tool-call format is chosen by the M1 probe under a rule fixed",
        "before the first run (PLAN M1 / ADR-0009). The suite is labelled",
        f"`{result.label}` and is **not** a headline number (n=20 → wide Wilson",
        "interval).",
        "",
        "## Decision rule",
        "",
        f"`native` becomes the default only if it is schema-valid on at least "
        f"{result.config.min_valid}/{result.config.n} tool-expecting prompts "
        "and its count is not lower than `json`'s; otherwise the default is "
        "`json`.",
        "",
        f"**Chosen default:** `{result.chosen_default}`",
        "",
        "## M1 tool-calling probe (smoke-only)",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Date | {meta.get('date', '')} |",
        f"| Model | {result.config.model} |",
        f"| Model GGUF sha256 | `{meta.get('model_gguf_sha256', '')}` |",
        f"| llama.cpp | {meta.get('llama_cpp_version', '')} |",
        f"| Launch flags | `{meta.get('launch_flags', '')}` |",
        f"| Temperature | {result.config.temperature} |",
        f"| Seed | {result.config.seed} |",
        f"| Thinking | enable_thinking={result.config.enable_thinking} |",
        f"| Freeze commit | `{result.freeze_commit_sha}` |",
        f"| Probe sha256 | `{result.probe_sha256}` |",
        f"| Head commit | `{result.head_commit_sha}` |",
        f"| Rule version | {result.config.rule_version} |",
        f"| Label | `{result.label}` |",
        "",
        "| Format | Schema-valid | 95% Wilson | No-tool correct | Parse failures |",
        "| --- | --- | --- | --- | --- |",
        (
            f"| native | {native.raw_count_label} | "
            f"[{native.wilson_lo:.3f}, {native.wilson_hi:.3f}] | "
            f"{native.no_tool_correct}/{native.n_no_tool} | {native.parse_failures} |"
        ),
        (
            f"| json | {json_fmt.raw_count_label} | "
            f"[{json_fmt.wilson_lo:.3f}, {json_fmt.wilson_hi:.3f}] | "
            f"{json_fmt.no_tool_correct}/{json_fmt.n_no_tool} | {json_fmt.parse_failures} |"
        ),
        "",
        "Note: with n=20 the Wilson interval is wide (18/20 ≈ [0.70, 0.97]).",
        'Passing the bar means "probably usable", not "90% reliable".',
        "Reliability is measured by `parse_failures` on full M5 runs.",
        "",
        "## Memory preflight",
        "",
        "```",
        result.doctor.get("formatted", "(skipped)"),
        "```",
        "",
    ]
    return "\n".join(lines)


def run_probe_cli() -> int:
    """CLI entry: guards, live run, write report + docs/providers.md."""
    import sys

    try:
        root = find_repo_root()
        result = asyncio.run(run_probe(repo_root=root))
    except ProbeRefuseError as exc:
        print(f"arp probe tool-calls: refused: {exc}", file=sys.stderr)
        return 1
    except ProbeError as exc:
        print(f"arp probe tool-calls: {exc}", file=sys.stderr)
        return 2

    report_path = write_report(result, root)
    providers_path = root / "docs" / "providers.md"
    providers_path.write_text(format_providers_md(result))

    native = result.formats["native"]
    json_fmt = result.formats["json"]
    print(f"probe sha256: {result.probe_sha256}")
    print(f"freeze commit: {result.freeze_commit_sha}")
    print(
        f"native: {native.raw_count_label} "
        f"Wilson95=[{native.wilson_lo:.3f}, {native.wilson_hi:.3f}]"
    )
    print(
        f"json:   {json_fmt.raw_count_label} "
        f"Wilson95=[{json_fmt.wilson_lo:.3f}, {json_fmt.wilson_hi:.3f}]"
    )
    print(f"chosen default: {result.chosen_default}")
    print(f"report: {report_path}")
    print(f"docs: {providers_path}")
    return 0
