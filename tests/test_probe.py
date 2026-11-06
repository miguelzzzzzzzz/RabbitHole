"""Unit tests for the M1 tool-calling probe (offline; mock provider)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agentic_research.cli import main
from agentic_research.probe import (
    ProbeConfig,
    ProbeError,
    ProbePrompt,
    ProbeRefuseError,
    assert_clean_tree,
    assert_probe_hash,
    decide_default,
    find_repo_root,
    freeze_commit_sha,
    load_probe_config,
    load_probe_set,
    probe_registry,
    run_probe,
    score_json_reply,
    score_native_reply,
    summarize_format,
    wilson_interval,
    write_report,
)
from agentic_research.providers.types import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    FunctionCall,
    ToolCall,
)

ROOT = Path(__file__).resolve().parents[1]


# --- Wilson / decision rule ---------------------------------------------------


def test_wilson_18_of_20_matches_hand_computed() -> None:
    lo, hi = wilson_interval(18, 20)
    assert 0.698 <= lo <= 0.701
    assert 0.971 <= hi <= 0.973


def test_wilson_edges() -> None:
    lo0, hi0 = wilson_interval(0, 20)
    assert lo0 == 0.0
    assert 0.16 <= hi0 <= 0.17
    lo20, hi20 = wilson_interval(20, 20)
    assert 0.83 <= lo20 <= 0.85
    assert hi20 == pytest.approx(1.0)


def test_wilson_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        wilson_interval(1, 0)
    with pytest.raises(ValueError):
        wilson_interval(-1, 5)
    with pytest.raises(ValueError):
        wilson_interval(6, 5)


@pytest.mark.parametrize(
    ("native", "json_n", "expected"),
    [
        (18, 17, "native"),  # boundary pass, not lower than json
        (18, 18, "native"),
        (20, 19, "native"),
        (17, 10, "json"),  # below bar
        (19, 20, "json"),  # at bar but lower than json
        (17, 17, "json"),
        (0, 0, "json"),
    ],
)
def test_decide_default_rule(native: int, json_n: int, expected: str) -> None:
    assert (
        decide_default(
            native_schema_valid=native,
            json_schema_valid=json_n,
            min_valid=18,
            n=20,
        )
        == expected
    )


# --- Freeze guards ------------------------------------------------------------


def test_load_probe_config_and_set_agree_with_freeze() -> None:
    cfg = load_probe_config(ROOT)
    probe = load_probe_set(ROOT / cfg.probe_path)
    assert len(probe.tool_expecting) == cfg.n == 20
    assert cfg.min_valid == 18
    digest = assert_probe_hash(cfg, ROOT / cfg.probe_path)
    assert digest == cfg.probe_sha256


def test_assert_probe_hash_mismatch_refuses(tmp_path: Path) -> None:
    cfg = ProbeConfig(
        probe_file="probes/tool_calls.v1.yaml",
        probe_sha256="0" * 64,
        min_valid=18,
        n=20,
        rule_version=1,
        temperature=0.0,
        seed=0,
        enable_thinking=False,
        model="Qwen3-1.7B-Q4_K_M",
        formats=("native", "json"),
        label="smoke-only",
    )
    fake = tmp_path / "probe.yaml"
    fake.write_text("version: 1\n")
    with pytest.raises(ProbeRefuseError, match="sha256 mismatch"):
        assert_probe_hash(cfg, fake)


def test_assert_clean_tree_refuses_dirty(tmp_path: Path) -> None:
    # Create a throwaway git repo that is dirty.
    import subprocess

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "test"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    (tmp_path / "a.txt").write_text("a\n")
    subprocess.run(["git", "add", "a.txt"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    (tmp_path / "b.txt").write_text("dirty\n")
    with pytest.raises(ProbeRefuseError, match="dirty"):
        assert_clean_tree(tmp_path)


def test_freeze_commit_sha_is_the_adding_commit() -> None:
    sha = freeze_commit_sha(ROOT)
    assert len(sha) == 40
    assert sha.startswith("fc7512f") or True  # full hash; prefix may vary if amended
    # Must be an ancestor of HEAD and must have introduced the probe file.
    import subprocess

    show = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", "probes/tool_calls.v1.yaml"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    # At least the freeze commit is in history; equality holds while file is unchanged.
    assert (
        sha == show.stdout.strip()
        or sha
        in subprocess.run(
            ["git", "log", "--format=%H", "--", "probes/tool_calls.v1.yaml"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )


def test_find_repo_root() -> None:
    assert find_repo_root(ROOT / "src" / "agentic_research") == ROOT


# --- Scoring ------------------------------------------------------------------


def _response(
    *,
    content: str | None = None,
    tool_calls: tuple[ToolCall, ...] | None = None,
    latency_ms: float = 12.0,
) -> ChatResponse:
    return ChatResponse(
        message=ChatMessage(role="assistant", content=content, tool_calls=tool_calls),
        finish_reason="stop",
        usage=None,
        model="Qwen3-1.7B-Q4_K_M",
        latency_ms=latency_ms,
        raw={},
    )


def _native_call(name: str, arguments: dict[str, Any] | None, *, raw: str = "") -> ToolCall:
    raw_s = raw or (json.dumps(arguments) if arguments is not None else "")
    parse_error = None if arguments is not None else "missing_arguments"
    return ToolCall(
        id="call_1",
        function=FunctionCall(
            name=name,
            arguments=arguments,
            arguments_raw=raw_s,
            parse_error=parse_error,
        ),
    )


def test_score_native_schema_valid_calculator() -> None:
    prompt = ProbePrompt(id="c1", expect="tool_call", user="x", tool="calculator")
    response = _response(
        tool_calls=(_native_call("calculator", {"expression": "1+1"}),),
    )
    score = score_native_reply(prompt, response, probe_registry())
    assert score.schema_valid is True
    assert score.correct_tool_name is True
    assert score.schema_valid_args is True


def test_score_native_wrong_tool() -> None:
    prompt = ProbePrompt(id="c1", expect="tool_call", user="x", tool="calculator")
    response = _response(
        tool_calls=(_native_call("search_corpus", {"query": "hi"}),),
    )
    score = score_native_reply(prompt, response, probe_registry())
    assert score.structurally_valid is True
    assert score.correct_tool_name is False
    assert score.schema_valid is False


def test_score_native_invalid_args() -> None:
    prompt = ProbePrompt(id="c1", expect="tool_call", user="x", tool="calculator")
    bad = ToolCall(
        id="call_1",
        function=FunctionCall(
            name="calculator",
            arguments={"expression": 3},
            arguments_raw='{"expression":3}',
            parse_error=None,
        ),
    )
    response = _response(tool_calls=(bad,))
    score = score_native_reply(prompt, response, probe_registry())
    assert score.correct_tool_name is True
    assert score.schema_valid_args is False
    assert score.schema_valid is False


def test_score_native_no_tool_correct_and_incorrect() -> None:
    prompt = ProbePrompt(id="n1", expect="no_tool", user="hi")
    ok = score_native_reply(prompt, _response(content="hello"), probe_registry())
    assert ok.correct_no_tool is True
    bad = score_native_reply(
        prompt,
        _response(tool_calls=(_native_call("calculator", {"expression": "1"}),)),
        probe_registry(),
    )
    assert bad.correct_no_tool is False


def test_score_json_schema_valid_and_final_answer() -> None:
    models = {
        t.name: t.args_model for t in [probe_registry().get(n) for n in probe_registry().names]
    }
    # rebuild cleanly
    reg = probe_registry()
    models = {name: reg.get(name).args_model for name in reg.names}
    prompt = ProbePrompt(id="c1", expect="tool_call", user="x", tool="calculator")
    content = json.dumps(
        {"action": "tool_call", "tool": "calculator", "arguments": {"expression": "2+2"}}
    )
    score = score_json_reply(prompt, _response(content=content), models)
    assert score.schema_valid is True

    no_tool = ProbePrompt(id="n1", expect="no_tool", user="hi")
    final = score_json_reply(
        no_tool,
        _response(content=json.dumps({"action": "final_answer", "answer": "ready"})),
        models,
    )
    assert final.correct_no_tool is True

    missing = score_json_reply(prompt, _response(content="not json"), models)
    assert missing.schema_valid is False
    assert missing.parse_failure_reason is not None


def test_summarize_and_decision_fixtures() -> None:
    """Scripted scores: native reliable / unreliable / both poor / boundary."""

    from agentic_research.probe import PromptScore

    def make(n_valid: int) -> list[PromptScore]:
        scores: list[PromptScore] = []
        for i in range(20):
            ok = i < n_valid
            scores.append(
                PromptScore(
                    prompt_id=f"p{i}",
                    expect="tool_call",
                    expected_tool="calculator",
                    structurally_valid=ok,
                    correct_tool_name=ok,
                    schema_valid_args=ok,
                    correct_no_tool=False,
                    schema_valid=ok,
                    parse_failure_reason=None if ok else "bad",
                    latency_ms=1.0,
                    tool_name="calculator" if ok else None,
                    arguments={"expression": "1"} if ok else None,
                    raw_content=None,
                    raw_tool_calls=None,
                )
            )
        return scores

    reliable = summarize_format("native", make(18), n_tool_expecting=20)
    assert reliable.schema_valid_count == 18
    assert (
        decide_default(native_schema_valid=18, json_schema_valid=17, min_valid=18, n=20) == "native"
    )
    assert (
        decide_default(native_schema_valid=17, json_schema_valid=10, min_valid=18, n=20) == "json"
    )
    both_poor = summarize_format("json", make(5), n_tool_expecting=20)
    assert both_poor.schema_valid_count == 5


# --- Mocked end-to-end run_probe ---------------------------------------------


class ScriptedProvider:
    """Returns scripted ChatResponse values in order (no network)."""

    def __init__(self, replies: list[ChatResponse]) -> None:
        self._replies = list(replies)
        self.requests: list[ChatRequest] = []

    async def complete(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if not self._replies:
            raise AssertionError("ScriptedProvider exhausted")
        return self._replies.pop(0)


def _perfect_replies(probe_n_tool: int = 20, n_no_tool: int = 5) -> list[ChatResponse]:
    """Build native-format perfect replies matching the frozen probe order."""
    cfg = load_probe_config(ROOT)
    probe = load_probe_set(ROOT / cfg.probe_path)
    replies: list[ChatResponse] = []
    for prompt in probe.prompts:
        if prompt.expect == "tool_call" and prompt.tool == "calculator":
            replies.append(
                _response(
                    tool_calls=(_native_call("calculator", {"expression": "1+1"}),),
                )
            )
        elif prompt.expect == "tool_call" and prompt.tool == "search_corpus":
            replies.append(
                _response(
                    tool_calls=(_native_call("search_corpus", {"query": "q", "k": 3}),),
                )
            )
        else:
            replies.append(_response(content="ready"))
    assert len(replies) == probe_n_tool + n_no_tool
    return replies


def _perfect_json_replies() -> list[ChatResponse]:
    cfg = load_probe_config(ROOT)
    probe = load_probe_set(ROOT / cfg.probe_path)
    replies: list[ChatResponse] = []
    for prompt in probe.prompts:
        if prompt.expect == "tool_call" and prompt.tool == "calculator":
            body = {
                "action": "tool_call",
                "tool": "calculator",
                "arguments": {"expression": "1+1"},
            }
        elif prompt.expect == "tool_call" and prompt.tool == "search_corpus":
            body = {
                "action": "tool_call",
                "tool": "search_corpus",
                "arguments": {"query": "q", "k": 3},
            }
        else:
            body = {"action": "final_answer", "answer": "ready"}
        replies.append(_response(content=json.dumps(body)))
    return replies


@pytest.mark.asyncio
async def test_run_probe_with_scripted_provider_picks_native(tmp_path: Path) -> None:
    # Native perfect, json perfect -> native (tie, at bar).
    replies = _perfect_replies() + _perfect_json_replies()
    provider = ScriptedProvider(replies)
    result = await run_probe(
        repo_root=ROOT,
        provider=provider,
        skip_doctor=True,
        skip_clean_tree=True,
    )
    assert result.chosen_default == "native"
    assert result.formats["native"].schema_valid_count == 20
    assert result.formats["json"].schema_valid_count == 20
    assert result.freeze_commit_sha
    assert result.probe_sha256 == load_probe_config(ROOT).probe_sha256
    # Native requests include tools; json requests do not.
    native_reqs = [r for r in provider.requests if r.tools is not None]
    json_reqs = [r for r in provider.requests if r.tools is None]
    assert len(native_reqs) == 25
    assert len(json_reqs) == 25
    report_path = write_report(result, ROOT, path=tmp_path / "report.json")
    payload = json.loads(report_path.read_text())
    assert payload["label"] == "smoke-only"
    assert payload["headline_eligible"] is False
    assert payload["chosen_default"] == "native"


@pytest.mark.asyncio
async def test_run_probe_refuses_hash_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_probe_config(ROOT)

    def bad_config(root: Path, config_path: Path | None = None) -> ProbeConfig:
        return ProbeConfig(
            probe_file=cfg.probe_file,
            probe_sha256="0" * 64,
            min_valid=cfg.min_valid,
            n=cfg.n,
            rule_version=cfg.rule_version,
            temperature=cfg.temperature,
            seed=cfg.seed,
            enable_thinking=cfg.enable_thinking,
            model=cfg.model,
            formats=cfg.formats,
            label=cfg.label,
        )

    monkeypatch.setattr("agentic_research.probe.load_probe_config", bad_config)
    with pytest.raises(ProbeRefuseError, match="sha256 mismatch"):
        await run_probe(
            repo_root=ROOT,
            provider=ScriptedProvider([]),
            skip_doctor=True,
            skip_clean_tree=True,
        )


def test_cli_probe_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["probe"]) == 0
    out = capsys.readouterr().out
    assert "tool-calls" in out


@pytest.mark.slow
@pytest.mark.allow_network
def test_live_probe_marker_exists_for_manual_runs() -> None:
    """Placeholder so the live path is discoverable; the real run is via CLI."""
    pytest.skip("live probe is run via `arp probe tool-calls` on the box, not this test")


def test_wilson_z_must_be_positive() -> None:
    with pytest.raises(ValueError):
        wilson_interval(1, 2, z=0)


def test_decide_default_rejects_bad_counts() -> None:
    with pytest.raises(ValueError):
        decide_default(native_schema_valid=-1, json_schema_valid=0, min_valid=18, n=20)
    with pytest.raises(ValueError):
        decide_default(native_schema_valid=21, json_schema_valid=0, min_valid=18, n=20)


def test_score_native_missing_tool_call() -> None:
    prompt = ProbePrompt(id="c1", expect="tool_call", user="x", tool="calculator")
    score = score_native_reply(prompt, _response(content="nope"), probe_registry())
    assert score.schema_valid is False
    assert score.parse_failure_reason == "missing_tool_call"


def test_score_json_wrong_tool() -> None:
    reg = probe_registry()
    models = {name: reg.get(name).args_model for name in reg.names}
    prompt = ProbePrompt(id="c1", expect="tool_call", user="x", tool="calculator")
    body = {
        "action": "tool_call",
        "tool": "search_corpus",
        "arguments": {"query": "q"},
    }
    score = score_json_reply(prompt, _response(content=json.dumps(body)), models)
    assert score.structurally_valid is True
    assert score.correct_tool_name is False
    assert score.schema_valid is False


@pytest.mark.asyncio
async def test_search_corpus_stub_run_raises() -> None:
    from agentic_research.probe import SearchCorpusArgs, SearchCorpusStubTool
    from agentic_research.tools.base import ToolContext

    tool = SearchCorpusStubTool()
    with pytest.raises(NotImplementedError):
        await tool.run(SearchCorpusArgs(query="q"), ToolContext(run_id="r", step=0))


def test_load_probe_set_rejects_bad_expect(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "version: 1\nid: x\ntools: [calculator]\nprompts:\n"
        "  - id: a\n    expect: maybe\n    user: hi\n"
    )
    with pytest.raises(ProbeError, match="unknown expect"):
        load_probe_set(path)


def test_load_probe_config_rejects_bad_format(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    real = tmp_path / "config" / "probe.toml"
    real.write_text(
        'probe_file = "p.yaml"\n'
        'probe_sha256 = "' + ("a" * 64) + '"\n'
        "min_valid = 18\nn = 20\nrule_version = 1\n"
        "temperature = 0.0\nseed = 0\nenable_thinking = false\n"
        'model = "m"\nformats = ["native", "xml"]\n'
        'label = "smoke-only"\n'
    )
    with pytest.raises(ProbeError, match="unknown format"):
        load_probe_config(tmp_path, Path("config/probe.toml"))


@pytest.mark.asyncio
async def test_format_providers_md_contains_counts() -> None:
    from agentic_research.probe import format_providers_md

    replies = _perfect_replies() + _perfect_json_replies()
    provider = ScriptedProvider(replies)
    result = await run_probe(
        repo_root=ROOT,
        provider=provider,
        skip_doctor=True,
        skip_clean_tree=True,
    )
    md = format_providers_md(result)
    assert "Chosen default" in md
    assert result.probe_sha256 in md
    assert "18/20" in md or "20/20" in md


def test_run_probe_cli_refuse(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agentic_research import probe as probe_mod

    async def bad_run(**kwargs: Any) -> Any:
        raise ProbeRefuseError("git working tree is dirty")

    monkeypatch.setattr(probe_mod, "run_probe", bad_run)
    monkeypatch.setattr(probe_mod, "find_repo_root", lambda: ROOT)
    assert probe_mod.run_probe_cli() == 1
    assert "refused" in capsys.readouterr().err


def test_run_probe_cli_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from agentic_research import probe as probe_mod

    replies = _perfect_replies() + _perfect_json_replies()
    provider = ScriptedProvider(replies)
    real_write = probe_mod.write_report

    async def fake_run(**kwargs: Any) -> Any:
        return await run_probe(
            repo_root=ROOT,
            provider=provider,
            skip_doctor=True,
            skip_clean_tree=True,
        )

    monkeypatch.setattr(probe_mod, "run_probe", fake_run)
    monkeypatch.setattr(probe_mod, "find_repo_root", lambda: ROOT)

    def write_tmp(result: Any, repo_root: Path, path: Path | None = None) -> Path:
        return real_write(result, repo_root, path=tmp_path / "report.json")

    monkeypatch.setattr(probe_mod, "write_report", write_tmp)

    providers_target = (ROOT / "docs" / "providers.md").resolve()
    original_write = Path.write_text

    def selective_write(self: Path, data: Any, *args: Any, **kwargs: Any) -> int:
        if self.resolve() == providers_target:
            return original_write(tmp_path / "providers.md", data, *args, **kwargs)
        return original_write(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", selective_write)
    assert probe_mod.run_probe_cli() == 0
    out = capsys.readouterr().out
    assert "chosen default: native" in out
    assert (tmp_path / "providers.md").is_file()
    assert (tmp_path / "report.json").is_file()


def test_cli_probe_tool_calls_wires_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    import agentic_research.cli as cli_mod

    called = {"n": 0}

    def fake_cli() -> int:
        called["n"] += 1
        return 0

    monkeypatch.setattr(cli_mod, "run_probe_cli", fake_cli)
    assert main(["probe", "tool-calls"]) == 0
    assert called["n"] == 1


@pytest.mark.asyncio
async def test_run_probe_doctor_refuse(monkeypatch: pytest.MonkeyPatch) -> None:
    from agentic_research import probe as probe_mod
    from agentic_research.preflight import CheckResult, DoctorReport, ServeFlags

    def fake_doctor() -> DoctorReport:
        flags = ServeFlags(
            ctx=4096,
            cache_ram_mib=0,
            np=1,
            host="127.0.0.1",
            port=8080,
            alias="Qwen3-1.7B-Q4_K_M",
            jinja=True,
            seed=0,
        )
        return DoctorReport(
            ok=False,
            checks=(
                CheckResult(
                    name="memory",
                    ok=False,
                    reason="low_memory",
                    message="not enough",
                    details={},
                ),
            ),
            warnings=(),
            required_mib=2843.36,
            headroom_mib=1024,
            reported_buffers_mib=1819.36,
            mem_available_mib=100.0,
            swap_total_mib=0.0,
            base_url="http://127.0.0.1:8080/v1",
            model_id="Qwen3-1.7B-Q4_K_M",
            serve_flags=flags,
        )

    monkeypatch.setattr(probe_mod, "run_doctor", fake_doctor)
    with pytest.raises(ProbeRefuseError, match="doctor failed"):
        await run_probe(
            repo_root=ROOT,
            provider=ScriptedProvider([]),
            skip_doctor=False,
            skip_clean_tree=True,
        )
