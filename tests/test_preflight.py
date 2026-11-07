"""Memory preflight and `arp doctor` checks (fake meminfo, mocked /v1/models)."""

from __future__ import annotations

import json
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

from agentic_research.cli import main
from agentic_research.preflight import (
    COMPUTE_BUFFER_MIB,
    DEFAULT_BASE_URL,
    DEFAULT_MIN_FREE_MB,
    KV_CACHE_MIB,
    REPORTED_SERVER_BUFFERS_MIB,
    REQUIRED_CACHE_RAM_MIB,
    REQUIRED_CTX,
    REQUIRED_MODEL_ALIAS,
    REQUIRED_REASONING_FORMAT,
    WEIGHTS_MMAP_MIB,
    WEIGHTS_REPACKED_MIB,
    CheckResult,
    DoctorReport,
    ObservedRuntime,
    ServeFlags,
    _check_serve_flags,
    base_url_from_environ,
    fetch_models,
    headroom_from_environ,
    intended_serve_flags,
    observe_runtime,
    parse_cmdline_flags,
    read_meminfo,
    required_ram_mib,
    run_doctor,
)

ROOT = Path(__file__).resolve().parents[1]
TCP_HEADER = (
    "  sl  local_address rem_address   st tx_queue rx_queue tr "
    "tm->when retrnsmt   uid  timeout inode\n"
)
TCP_LISTEN_8080 = (
    "   0: 0100007F:1F90 00000000:0000 0A 00000000:00000000 "
    "00:00000000 00000000     0        0 555 1 "
    "0000000000000000 100 0 0 10 0\n"
)
TCP_LISTEN_8080_INODE77 = (
    "   0: 0100007F:1F90 00000000:0000 0A 00000000:00000000 "
    "00:00000000 00000000     0        0 77 1 "
    "0000000000000000 100 0 0 10 0\n"
)
DOC = ROOT / "docs" / "local-models.md"
SCRIPT = ROOT / "scripts" / "local_runtime.sh"


def _meminfo_text(*, available_kib: int, swap_total_kib: int = 0) -> str:
    return (
        f"MemTotal:       {available_kib + 1_000_000} kB\n"
        f"MemFree:        {available_kib // 2} kB\n"
        f"MemAvailable:   {available_kib} kB\n"
        f"SwapTotal:      {swap_total_kib} kB\n"
        f"SwapFree:       {swap_total_kib} kB\n"
    )


def _write_meminfo(path: Path, *, available_kib: int, swap_total_kib: int = 0) -> Path:
    path.write_text(
        _meminfo_text(available_kib=available_kib, swap_total_kib=swap_total_kib),
        encoding="utf-8",
    )
    return path


def _models_payload(*ids: str) -> dict[str, Any]:
    return {"object": "list", "data": [{"id": model_id, "object": "model"} for model_id in ids]}


def _pass_env() -> dict[str, str]:
    return {
        "ARP_LOCAL_CTX": str(REQUIRED_CTX),
        "ARP_LOCAL_MODEL": REQUIRED_MODEL_ALIAS,
        "ARP_LLM_BASE_URL": DEFAULT_BASE_URL,
        "ARP_MIN_FREE_MB": str(DEFAULT_MIN_FREE_MB),
    }


# --- constants stay in sync with docs / script --------------------------------


def test_reported_buffers_sum_matches_constant() -> None:
    assert (
        pytest.approx(WEIGHTS_REPACKED_MIB + WEIGHTS_MMAP_MIB + KV_CACHE_MIB + COMPUTE_BUFFER_MIB)
        == REPORTED_SERVER_BUFFERS_MIB
    )
    assert pytest.approx(1819.36) == REPORTED_SERVER_BUFFERS_MIB


def test_doc_records_each_reported_buffer() -> None:
    doc = DOC.read_text(encoding="utf-8")
    for value in (
        WEIGHTS_REPACKED_MIB,
        WEIGHTS_MMAP_MIB,
        KV_CACHE_MIB,
        COMPUTE_BUFFER_MIB,
    ):
        assert f"{value:.2f}" in doc or f"{value:.0f}" in doc or str(value) in doc


def test_script_serve_flags_match_required_pins() -> None:
    serve = SCRIPT.read_text(encoding="utf-8").split("serve() {", 1)[1].split("\n}\n", 1)[0]
    assert '-c "$CTX"' in serve or '-c "$CTX"' in serve
    assert f"--cache-ram {REQUIRED_CACHE_RAM_MIB}" in serve
    assert "-np 1" in serve
    assert "--jinja" in serve
    assert "--seed 0" in serve
    assert "--reasoning-format none" in serve
    assert 'CTX="${ARP_LOCAL_CTX:-4096}"' in SCRIPT.read_text(encoding="utf-8")


def test_required_ram_includes_headroom() -> None:
    assert required_ram_mib(1024) == pytest.approx(REPORTED_SERVER_BUFFERS_MIB + 1024)
    assert required_ram_mib(0) == pytest.approx(REPORTED_SERVER_BUFFERS_MIB)
    with pytest.raises(ValueError, match="headroom_mib"):
        required_ram_mib(-1)


# --- meminfo ------------------------------------------------------------------


def test_read_meminfo_parses_available_and_swap(tmp_path: Path) -> None:
    path = _write_meminfo(tmp_path / "meminfo", available_kib=3_145_728, swap_total_kib=0)
    mem = read_meminfo(path)
    assert mem.mem_available_kib == 3_145_728
    assert mem.swap_total_kib == 0
    assert mem.mem_available_mib == pytest.approx(3072.0)


def test_read_meminfo_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "meminfo"
    path.write_text("MemTotal: 1000 kB\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing MemAvailable"):
        read_meminfo(path)


# --- serve flags --------------------------------------------------------------


def test_intended_serve_flags_defaults() -> None:
    flags = intended_serve_flags({})
    assert flags == ServeFlags(
        ctx=REQUIRED_CTX,
        cache_ram_mib=REQUIRED_CACHE_RAM_MIB,
        np=1,
        host="127.0.0.1",
        port=8080,
        alias=REQUIRED_MODEL_ALIAS,
        jinja=True,
        seed=0,
        reasoning_format=REQUIRED_REASONING_FORMAT,
    )
    assert flags.ok


def test_intended_serve_flags_rejects_non_integer_ctx() -> None:
    with pytest.raises(ValueError, match="ARP_LOCAL_CTX"):
        intended_serve_flags({"ARP_LOCAL_CTX": "nope"})


def test_headroom_and_base_url_parsing() -> None:
    assert headroom_from_environ({}) == DEFAULT_MIN_FREE_MB
    assert headroom_from_environ({"ARP_MIN_FREE_MB": "512"}) == 512
    with pytest.raises(ValueError, match="ARP_MIN_FREE_MB"):
        headroom_from_environ({"ARP_MIN_FREE_MB": "x"})
    with pytest.raises(ValueError, match="ARP_MIN_FREE_MB"):
        headroom_from_environ({"ARP_MIN_FREE_MB": "-1"})
    assert base_url_from_environ({}) == DEFAULT_BASE_URL
    assert base_url_from_environ({"ARP_LLM_BASE_URL": "http://127.0.0.1:9/v1/"}) == (
        "http://127.0.0.1:9/v1"
    )


# --- run_doctor ---------------------------------------------------------------


def test_doctor_passes_above_threshold(tmp_path: Path) -> None:
    # required ~1819.36 + 1024 = ~2843.36 MiB -> need >= that many KiB
    need_kib = int(required_ram_mib(DEFAULT_MIN_FREE_MB) * 1024) + 10_000
    path = _write_meminfo(tmp_path / "meminfo", available_kib=need_kib, swap_total_kib=0)
    report = run_doctor(
        environ=_pass_env(),
        meminfo_path=path,
        fetch_models_fn=lambda _url: _models_payload(REQUIRED_MODEL_ALIAS),
        skip_observe=True,
    )
    assert report.ok
    assert report.swap_total_mib == 0.0
    by_name = {c.name: c for c in report.checks}
    assert by_name["serve_flags"].ok
    assert by_name["memory"].ok
    assert by_name["models"].ok
    assert "pass" in report.format()


def test_doctor_refuses_below_threshold(tmp_path: Path) -> None:
    # Well under the required budget.
    path = _write_meminfo(tmp_path / "meminfo", available_kib=500_000, swap_total_kib=0)
    report = run_doctor(
        environ=_pass_env(),
        meminfo_path=path,
        fetch_models_fn=lambda _url: _models_payload(REQUIRED_MODEL_ALIAS),
        skip_observe=True,
    )
    assert not report.ok
    memory = next(c for c in report.checks if c.name == "memory")
    assert memory.reason == "insufficient_memory"
    assert "swap is not counted" in memory.message
    assert "refuse" in report.format()


def test_doctor_boundary_just_enough(tmp_path: Path) -> None:
    required = required_ram_mib(DEFAULT_MIN_FREE_MB)
    # Exact boundary in KiB (floor of required * 1024 would be short; use ceil).
    import math

    available_kib = math.ceil(required * 1024)
    path = _write_meminfo(tmp_path / "meminfo", available_kib=available_kib)
    report = run_doctor(
        environ=_pass_env(),
        meminfo_path=path,
        fetch_models_fn=lambda _url: _models_payload(REQUIRED_MODEL_ALIAS),
        skip_observe=True,
    )
    assert report.ok
    memory = next(c for c in report.checks if c.name == "memory")
    assert memory.ok


def test_doctor_boundary_one_kib_short(tmp_path: Path) -> None:
    import math

    required = required_ram_mib(DEFAULT_MIN_FREE_MB)
    available_kib = math.ceil(required * 1024) - 1
    path = _write_meminfo(tmp_path / "meminfo", available_kib=available_kib)
    report = run_doctor(
        environ=_pass_env(),
        meminfo_path=path,
        fetch_models_fn=lambda _url: _models_payload(REQUIRED_MODEL_ALIAS),
        skip_observe=True,
    )
    assert not report.ok
    assert next(c for c in report.checks if c.name == "memory").reason == "insufficient_memory"


def test_doctor_reports_swap_but_does_not_count_it(tmp_path: Path) -> None:
    # Plenty of MemAvailable, but swap present -> warn, still pass.
    need_kib = int(required_ram_mib(DEFAULT_MIN_FREE_MB) * 1024) + 50_000
    path = _write_meminfo(tmp_path / "meminfo", available_kib=need_kib, swap_total_kib=2_097_152)
    report = run_doctor(
        environ=_pass_env(),
        meminfo_path=path,
        fetch_models_fn=lambda _url: _models_payload(REQUIRED_MODEL_ALIAS),
        skip_observe=True,
    )
    assert report.ok
    assert report.swap_total_mib == pytest.approx(2048.0)
    assert any("SwapTotal" in w for w in report.warnings)
    assert "[warn]" in report.format()


def test_doctor_ignores_swap_when_mem_short(tmp_path: Path) -> None:
    # Huge swap, tiny MemAvailable -> still refuse.
    path = _write_meminfo(tmp_path / "meminfo", available_kib=100_000, swap_total_kib=50_000_000)
    report = run_doctor(
        environ=_pass_env(),
        meminfo_path=path,
        fetch_models_fn=lambda _url: _models_payload(REQUIRED_MODEL_ALIAS),
        skip_observe=True,
    )
    assert not report.ok
    assert next(c for c in report.checks if c.name == "memory").reason == "insufficient_memory"


@pytest.mark.parametrize(
    ("environ", "needle"),
    [
        ({"ARP_LOCAL_CTX": "2048"}, "context is 2048"),
        ({"ARP_LOCAL_MODEL": "other"}, "alias is 'other'"),
    ],
)
def test_doctor_refuses_bad_serve_flags(
    tmp_path: Path, environ: dict[str, str], needle: str
) -> None:
    need_kib = int(required_ram_mib(DEFAULT_MIN_FREE_MB) * 1024) + 10_000
    path = _write_meminfo(tmp_path / "meminfo", available_kib=need_kib)
    env = {**_pass_env(), **environ}
    report = run_doctor(
        environ=env,
        meminfo_path=path,
        fetch_models_fn=lambda _url: _models_payload(env["ARP_LOCAL_MODEL"]),
        skip_observe=True,
    )
    assert not report.ok
    flags = next(c for c in report.checks if c.name == "serve_flags")
    assert flags.reason == "bad_serve_flags"
    assert needle in flags.message


def test_doctor_refuses_when_models_unreachable(tmp_path: Path) -> None:
    need_kib = int(required_ram_mib(DEFAULT_MIN_FREE_MB) * 1024) + 10_000
    path = _write_meminfo(tmp_path / "meminfo", available_kib=need_kib)

    def boom(_url: str) -> dict[str, Any]:
        raise OSError("connection refused")

    report = run_doctor(
        environ=_pass_env(),
        meminfo_path=path,
        fetch_models_fn=boom,
        skip_observe=True,
    )
    assert not report.ok
    models = next(c for c in report.checks if c.name == "models")
    assert models.reason == "local_server_unavailable"


def test_doctor_refuses_when_model_id_missing(tmp_path: Path) -> None:
    need_kib = int(required_ram_mib(DEFAULT_MIN_FREE_MB) * 1024) + 10_000
    path = _write_meminfo(tmp_path / "meminfo", available_kib=need_kib)
    report = run_doctor(
        environ=_pass_env(),
        meminfo_path=path,
        fetch_models_fn=lambda _url: _models_payload("someone-else"),
        skip_observe=True,
    )
    assert not report.ok
    assert next(c for c in report.checks if c.name == "models").reason == "model_missing"


def test_doctor_refuses_bad_models_shape(tmp_path: Path) -> None:
    need_kib = int(required_ram_mib(DEFAULT_MIN_FREE_MB) * 1024) + 10_000
    path = _write_meminfo(tmp_path / "meminfo", available_kib=need_kib)
    report = run_doctor(
        environ=_pass_env(),
        meminfo_path=path,
        fetch_models_fn=lambda _url: {"object": "list"},
        skip_observe=True,
    )
    assert not report.ok
    assert next(c for c in report.checks if c.name == "models").reason == "bad_models_response"


def test_doctor_meminfo_unreadable(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    report = run_doctor(
        environ=_pass_env(),
        meminfo_path=missing,
        fetch_models_fn=lambda _url: _models_payload(REQUIRED_MODEL_ALIAS),
        skip_observe=True,
    )
    assert not report.ok
    assert next(c for c in report.checks if c.name == "memory").reason == "meminfo_unreadable"


def test_doctor_custom_headroom(tmp_path: Path) -> None:
    # With headroom 0, a modest MemAvailable that would fail the default passes.
    available_kib = int((REPORTED_SERVER_BUFFERS_MIB + 50) * 1024)
    path = _write_meminfo(tmp_path / "meminfo", available_kib=available_kib)
    env = {**_pass_env(), "ARP_MIN_FREE_MB": "0"}
    report = run_doctor(
        environ=env,
        meminfo_path=path,
        fetch_models_fn=lambda _url: _models_payload(REQUIRED_MODEL_ALIAS),
        skip_observe=True,
    )
    assert report.ok
    assert report.headroom_mib == 0


def test_fetch_models_uses_injected_opener(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise fetch_models without opening a real socket (network guard on)."""

    class _Resp:
        def read(self) -> bytes:
            return json.dumps(_models_payload(REQUIRED_MODEL_ALIAS)).encode()

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    calls: list[str] = []

    def fake_urlopen(request: Any, timeout: float = 0) -> _Resp:
        calls.append(request.get_full_url())
        assert timeout == 5.0
        return _Resp()

    monkeypatch.setattr("agentic_research.preflight.urllib.request.urlopen", fake_urlopen)
    payload = fetch_models(DEFAULT_BASE_URL)
    assert payload["data"][0]["id"] == REQUIRED_MODEL_ALIAS
    assert calls == [f"{DEFAULT_BASE_URL}/models"]


def test_fetch_models_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error

    def boom(request: Any, timeout: float = 0) -> Any:
        raise urllib.error.HTTPError(
            request.get_full_url(),
            500,
            "nope",
            Message(),
            None,
        )

    monkeypatch.setattr("agentic_research.preflight.urllib.request.urlopen", boom)
    with pytest.raises(OSError, match="HTTP 500"):
        fetch_models(DEFAULT_BASE_URL)


def test_fetch_models_url_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error

    def boom(request: Any, timeout: float = 0) -> Any:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("agentic_research.preflight.urllib.request.urlopen", boom)
    with pytest.raises(OSError, match="cannot reach"):
        fetch_models(DEFAULT_BASE_URL)


def test_fetch_models_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        def read(self) -> bytes:
            return b"not-json"

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(
        "agentic_research.preflight.urllib.request.urlopen", lambda *a, **k: _Resp()
    )
    with pytest.raises(OSError, match="invalid JSON"):
        fetch_models(DEFAULT_BASE_URL)


def test_fetch_models_non_object(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        def read(self) -> bytes:
            return b"[1]"

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(
        "agentic_research.preflight.urllib.request.urlopen", lambda *a, **k: _Resp()
    )
    with pytest.raises(OSError, match="JSON object"):
        fetch_models(DEFAULT_BASE_URL)


# --- CLI ----------------------------------------------------------------------


def test_arp_doctor_cli_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    need_kib = int(required_ram_mib(DEFAULT_MIN_FREE_MB) * 1024) + 10_000
    path = _write_meminfo(tmp_path / "meminfo", available_kib=need_kib)
    monkeypatch.setenv("ARP_LOCAL_CTX", str(REQUIRED_CTX))
    monkeypatch.setenv("ARP_LOCAL_MODEL", REQUIRED_MODEL_ALIAS)
    monkeypatch.setenv("ARP_LLM_BASE_URL", DEFAULT_BASE_URL)
    monkeypatch.setenv("ARP_MIN_FREE_MB", str(DEFAULT_MIN_FREE_MB))

    def fake_run_doctor(**kwargs: Any) -> DoctorReport:
        return run_doctor(
            environ=_pass_env(),
            meminfo_path=path,
            fetch_models_fn=lambda _url: _models_payload(REQUIRED_MODEL_ALIAS),
            skip_observe=True,
        )

    monkeypatch.setattr("agentic_research.cli.run_doctor", fake_run_doctor)
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "pass" in out
    assert "[ok] memory:" in out
    assert "[ok] models:" in out


def test_arp_doctor_cli_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write_meminfo(tmp_path / "meminfo", available_kib=1000)

    def fake_run_doctor(**kwargs: Any) -> DoctorReport:
        return run_doctor(
            environ=_pass_env(),
            meminfo_path=path,
            fetch_models_fn=lambda _url: _models_payload(REQUIRED_MODEL_ALIAS),
            skip_observe=True,
        )

    monkeypatch.setattr("agentic_research.cli.run_doctor", fake_run_doctor)
    assert main(["doctor"]) == 1
    assert "refuse" in capsys.readouterr().out


def test_read_meminfo_skips_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "meminfo"
    path.write_text(
        "\n"
        "no_colon_here\n"
        "EmptyValue:\n"
        "NotANumber: abc kB\n"
        "MemAvailable: 2048 kB\n"
        "SwapTotal: 0 kB\n",
        encoding="utf-8",
    )
    mem = read_meminfo(path)
    assert mem.mem_available_kib == 2048
    assert mem.swap_total_kib == 0


def test_check_serve_flags_reports_every_mismatch() -> None:
    flags = ServeFlags(
        ctx=2048,
        cache_ram_mib=8192,
        np=2,
        host="0.0.0.0",
        port=9999,
        alias="wrong",
        jinja=False,
        seed=7,
        reasoning_format="auto",
    )
    result = _check_serve_flags(flags, observed=True)
    assert not result.ok
    assert result.reason == "bad_serve_flags"
    for needle in (
        "context is 2048",
        "--cache-ram is 8192",
        "-np is 2",
        "bind is 0.0.0.0:9999",
        "alias is 'wrong'",
        "--jinja is required",
        "--seed is 7",
        "--reasoning-format is 'auto'",
    ):
        assert needle in result.message


def test_format_without_meminfo_and_without_jinja() -> None:
    flags = ServeFlags(
        ctx=REQUIRED_CTX,
        cache_ram_mib=REQUIRED_CACHE_RAM_MIB,
        np=1,
        host="127.0.0.1",
        port=8080,
        alias=REQUIRED_MODEL_ALIAS,
        jinja=False,
        seed=0,
        reasoning_format=REQUIRED_REASONING_FORMAT,
    )
    report = DoctorReport(
        ok=False,
        checks=(
            CheckResult(
                name="memory",
                ok=False,
                reason="meminfo_unreadable",
                message="cannot read",
                details={},
            ),
        ),
        warnings=(),
        required_mib=required_ram_mib(),
        headroom_mib=DEFAULT_MIN_FREE_MB,
        reported_buffers_mib=REPORTED_SERVER_BUFFERS_MIB,
        mem_available_mib=None,
        swap_total_mib=None,
        base_url=DEFAULT_BASE_URL,
        model_id=REQUIRED_MODEL_ALIAS,
        serve_flags=flags,
    )
    text = report.format()
    assert "MemAvailable:" not in text
    assert "--jinja" not in text.split("serve flags (observed):", 1)[1].splitlines()[0]
    assert "refuse" in text


def test_doctor_skips_models_when_requested(tmp_path: Path) -> None:
    need_kib = int(required_ram_mib(DEFAULT_MIN_FREE_MB) * 1024) + 10_000
    path = _write_meminfo(tmp_path / "meminfo", available_kib=need_kib)
    report = run_doctor(environ=_pass_env(), meminfo_path=path, skip_models=True, skip_observe=True)
    assert report.ok
    models = next(c for c in report.checks if c.name == "models")
    assert models.reason == "skipped"


def test_doctor_ignores_non_model_entries(tmp_path: Path) -> None:
    need_kib = int(required_ram_mib(DEFAULT_MIN_FREE_MB) * 1024) + 10_000
    path = _write_meminfo(tmp_path / "meminfo", available_kib=need_kib)

    def payload(_url: str) -> dict[str, Any]:
        return {
            "data": [
                "skip-me",
                {"id": 123},
                {"id": REQUIRED_MODEL_ALIAS},
            ]
        }

    report = run_doctor(
        environ=_pass_env(),
        meminfo_path=path,
        fetch_models_fn=payload,
        skip_observe=True,
    )
    assert report.ok


def test_doctor_format_includes_mem_when_unreadable_path_not_used(
    tmp_path: Path,
) -> None:
    # Cover the meminfo_unreadable path through format() as well.
    report = run_doctor(
        environ=_pass_env(),
        meminfo_path=tmp_path / "missing",
        fetch_models_fn=lambda _url: _models_payload(REQUIRED_MODEL_ALIAS),
        skip_observe=True,
    )
    text = report.format()
    assert "MemAvailable:" not in text
    assert "meminfo_unreadable" not in text  # reason is not printed; message is
    assert "cannot read meminfo" in text


# --- live observation (MAJOR-1) ----------------------------------------------


def _good_cmdline(*, cache_ram: int = 0, reasoning: str = "none") -> list[str]:
    return [
        "/workspace/portfolio/runtime/llama-b12032/llama-server",
        "-m",
        "/workspace/portfolio/runtime/models/Qwen3-1.7B-Q4_K_M.gguf",
        "--alias",
        REQUIRED_MODEL_ALIAS,
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
        "-c",
        "4096",
        "-np",
        "1",
        "--jinja",
        "--cache-ram",
        str(cache_ram),
        "--seed",
        "0",
        "--reasoning-format",
        reasoning,
    ]


def test_parse_cmdline_flags_defaults_missing_cache_and_reasoning() -> None:
    flags = parse_cmdline_flags(
        ["llama-server", "-m", "x.gguf", "--alias", REQUIRED_MODEL_ALIAS, "-c", "4096", "-np", "1"]
    )
    assert flags.cache_ram_mib == 8192
    assert flags.reasoning_format == "auto"
    assert not flags.ok


def test_doctor_refuses_when_observed_cache_ram_on(tmp_path: Path) -> None:
    need_kib = int(required_ram_mib(DEFAULT_MIN_FREE_MB) * 1024) + 10_000
    path = _write_meminfo(tmp_path / "meminfo", available_kib=need_kib)

    def fake_observe(*, base_url: str = "", **_kwargs: object) -> ObservedRuntime:
        argv = _good_cmdline(cache_ram=8192)
        flags = parse_cmdline_flags(argv)
        return ObservedRuntime(
            pid=4242,
            cmdline=tuple(argv),
            flags=flags,
            props=None,
            model_sha256=None,
            llama_version=None,
            binary_path=argv[0],
        )

    report = run_doctor(
        environ=_pass_env(),
        meminfo_path=path,
        fetch_models_fn=lambda _url: _models_payload(REQUIRED_MODEL_ALIAS),
        observe_fn=fake_observe,
    )
    assert not report.ok
    flags = next(c for c in report.checks if c.name == "serve_flags")
    assert flags.reason == "bad_serve_flags"
    assert "--cache-ram is 8192" in flags.message
    assert flags.details["source"] == "observed"


def test_doctor_refuses_when_serve_flags_unreadable(tmp_path: Path) -> None:
    need_kib = int(required_ram_mib(DEFAULT_MIN_FREE_MB) * 1024) + 10_000
    path = _write_meminfo(tmp_path / "meminfo", available_kib=need_kib)

    def fake_observe(*, base_url: str = "", **_kwargs: object) -> ObservedRuntime:
        return ObservedRuntime(
            pid=-1,
            cmdline=(),
            flags=ServeFlags(
                ctx=-1,
                cache_ram_mib=-1,
                np=-1,
                host="",
                port=8080,
                alias="",
                jinja=False,
                seed=-1,
                reasoning_format="unreadable",
            ),
            props=None,
            model_sha256=None,
            llama_version=None,
            binary_path=None,
            observation_error="no process listening on port 8080",
        )

    report = run_doctor(
        environ=_pass_env(),
        meminfo_path=path,
        fetch_models_fn=lambda _url: _models_payload(REQUIRED_MODEL_ALIAS),
        observe_fn=fake_observe,
    )
    assert not report.ok
    flags = next(c for c in report.checks if c.name == "serve_flags")
    assert flags.reason == "serve_flags_unreadable"


def test_doctor_passes_with_good_observed_cmdline(tmp_path: Path) -> None:
    need_kib = int(required_ram_mib(DEFAULT_MIN_FREE_MB) * 1024) + 10_000
    path = _write_meminfo(tmp_path / "meminfo", available_kib=need_kib)

    def fake_observe(*, base_url: str = "", **_kwargs: object) -> ObservedRuntime:
        argv = _good_cmdline()
        flags = parse_cmdline_flags(argv)
        return ObservedRuntime(
            pid=7,
            cmdline=tuple(argv),
            flags=flags,
            props={
                "model_path": flags.model_path,
                "model_alias": REQUIRED_MODEL_ALIAS,
                "total_slots": 1,
                "default_generation_settings": {
                    "n_ctx": 4096,
                    "params": {"reasoning_format": "none"},
                },
            },
            model_sha256="b139949c5bd74937ad8ed8c8cf3d9ffb1e99c866c823204dc42c0d91fa181897",
            llama_version="0.8.0-dev (build 12032, commit dde879bc4)",
            binary_path=argv[0],
        )

    report = run_doctor(
        environ=_pass_env(),
        meminfo_path=path,
        fetch_models_fn=lambda _url: _models_payload(REQUIRED_MODEL_ALIAS),
        observe_fn=fake_observe,
    )
    assert report.ok
    assert report.observed is not None
    assert report.observed.pid == 7
    assert "observed" in next(c for c in report.checks if c.name == "serve_flags").message


def test_observe_runtime_accepts_injected_cmdline() -> None:
    argv = _good_cmdline(cache_ram=0)
    obs = observe_runtime(
        base_url=DEFAULT_BASE_URL,
        cmdline=argv,
        pid=99,
        fetch_props_fn=lambda _url: None,
        hash_model=False,
        read_version=False,
    )
    assert obs.observation_error is None
    assert obs.flags.ok
    assert obs.pid == 99


# --- /proc observation helpers coverage --------------------------------------


def test_port_and_server_root_helpers() -> None:
    from agentic_research.preflight import port_from_base_url, server_root_from_base_url

    assert port_from_base_url("http://127.0.0.1:8080/v1") == 8080
    assert port_from_base_url("https://example.com/v1") == 443
    assert port_from_base_url("http://example.com/v1") == 80
    assert server_root_from_base_url("http://127.0.0.1:8080/v1") == "http://127.0.0.1:8080"
    assert server_root_from_base_url("http://127.0.0.1:8080") == "http://127.0.0.1:8080"


def test_find_listener_pid_from_fake_proc(tmp_path: Path) -> None:
    from agentic_research.preflight import find_listener_pid, read_cmdline

    proc = tmp_path / "proc"
    net = proc / "net"
    net.mkdir(parents=True)
    (net / "tcp").write_text(TCP_HEADER + TCP_LISTEN_8080, encoding="utf-8")
    (net / "tcp6").write_text(TCP_HEADER, encoding="utf-8")
    pid_dir = proc / "4242"
    (pid_dir / "fd").mkdir(parents=True)
    (pid_dir / "cmdline").write_bytes(b"llama-server\0-c\0" + b"4096\0")
    sock = pid_dir / "fd" / "3"
    sock.symlink_to("socket:[555]")

    assert find_listener_pid(8080, proc_root=proc) == 4242
    assert find_listener_pid(9999, proc_root=proc) is None
    assert read_cmdline(4242, proc_root=proc) == ["llama-server", "-c", "4096"]


def test_observe_runtime_no_listener(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    (proc / "net").mkdir(parents=True)
    (proc / "net" / "tcp").write_text(TCP_HEADER, encoding="utf-8")
    obs = observe_runtime(
        base_url=DEFAULT_BASE_URL,
        proc_root=proc,
        fetch_props_fn=lambda _url: None,
        hash_model=False,
        read_version=False,
    )
    assert obs.observation_error is not None
    assert "no process listening" in obs.observation_error


def test_observe_runtime_empty_cmdline(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    net = proc / "net"
    net.mkdir(parents=True)
    (net / "tcp").write_text(TCP_HEADER + TCP_LISTEN_8080_INODE77, encoding="utf-8")
    pid_dir = proc / "9"
    (pid_dir / "fd").mkdir(parents=True)
    (pid_dir / "cmdline").write_bytes(b"")
    (pid_dir / "fd" / "1").symlink_to("socket:[77]")
    obs = observe_runtime(
        base_url=DEFAULT_BASE_URL,
        proc_root=proc,
        fetch_props_fn=lambda _url: None,
        hash_model=False,
        read_version=False,
    )
    assert obs.observation_error is not None
    assert "cmdline" in obs.observation_error


def test_parse_cmdline_equals_form_and_no_jinja() -> None:
    flags = parse_cmdline_flags(
        [
            "llama-server",
            "--ctx-size=4096",
            "--cache-ram=0",
            "--parallel=1",
            "--host=127.0.0.1",
            "--port=8080",
            "--alias=Qwen3-1.7B-Q4_K_M",
            "--no-jinja",
            "--seed=0",
            "--reasoning-format=none",
            "-m",
            "/tmp/model.gguf",
        ]
    )
    assert flags.ctx == 4096
    assert flags.cache_ram_mib == 0
    assert flags.np == 1
    assert flags.jinja is False
    assert flags.model_path == "/tmp/model.gguf"
    assert flags.reasoning_format == "none"


def test_sha256_file_and_launch_summary(tmp_path: Path) -> None:
    from agentic_research.preflight import sha256_file

    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    assert sha256_file(f) == ("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    flags = intended_serve_flags({})
    assert "--reasoning-format none" in flags.launch_flags_summary()


def test_fetch_props_unreachable_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.request

    from agentic_research.preflight import fetch_props

    def boom(*_a: object, **_k: object) -> object:
        raise urllib.error.URLError("nope")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert fetch_props("http://127.0.0.1:9/v1") is None


def test_llama_server_version_missing_binary(tmp_path: Path) -> None:
    from agentic_research.preflight import llama_server_version

    assert llama_server_version(tmp_path / "missing") is None


def test_observe_fills_from_props_when_cmdline_sparse() -> None:
    argv = [
        "llama-server",
        "--cache-ram",
        "0",
        "--reasoning-format",
        "none",
        "--jinja",
        "--seed",
        "0",
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
    ]
    props = {
        "model_path": "/tmp/m.gguf",
        "model_alias": REQUIRED_MODEL_ALIAS,
        "total_slots": 1,
        "default_generation_settings": {"n_ctx": 4096, "params": {"reasoning_format": "none"}},
    }
    obs = observe_runtime(
        base_url=DEFAULT_BASE_URL,
        cmdline=argv,
        pid=1,
        fetch_props_fn=lambda _url: props,
        hash_model=False,
        read_version=False,
    )
    assert obs.flags.model_path == "/tmp/m.gguf"
    assert obs.flags.alias == REQUIRED_MODEL_ALIAS
    assert obs.flags.ctx == 4096
    assert obs.flags.np == 1


def test_observe_hashes_model_and_reads_version(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf-bytes")
    binary = tmp_path / "llama-server"
    binary.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then\n'
        "  echo 'version: 0.8.0-dev (build 12032, commit dde879bc4)'\n"
        "  echo 'built with GNU'\n"
        "fi\n"
    )
    binary.chmod(0o755)
    argv = [
        str(binary),
        "-m",
        str(model),
        "--alias",
        REQUIRED_MODEL_ALIAS,
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
        "-c",
        "4096",
        "-np",
        "1",
        "--jinja",
        "--cache-ram",
        "0",
        "--seed",
        "0",
        "--reasoning-format",
        "none",
    ]
    obs = observe_runtime(
        base_url=DEFAULT_BASE_URL,
        cmdline=argv,
        pid=3,
        fetch_props_fn=lambda _url: None,
        hash_model=True,
        read_version=True,
    )
    assert obs.model_sha256 is not None
    assert len(obs.model_sha256) == 64
    assert obs.llama_version is not None
    assert "12032" in obs.llama_version


def test_doctor_format_includes_observed_sha(tmp_path: Path) -> None:
    need_kib = int(required_ram_mib(DEFAULT_MIN_FREE_MB) * 1024) + 10_000
    path = _write_meminfo(tmp_path / "meminfo", available_kib=need_kib)

    def fake_observe(*, base_url: str = "", **_k: object) -> ObservedRuntime:
        argv = _good_cmdline()
        return ObservedRuntime(
            pid=11,
            cmdline=tuple(argv),
            flags=parse_cmdline_flags(argv),
            props=None,
            model_sha256="a" * 64,
            llama_version="0.8.0-dev (build 12032, commit dde879bc4)",
            binary_path=argv[0],
        )

    report = run_doctor(
        environ=_pass_env(),
        meminfo_path=path,
        fetch_models_fn=lambda _url: _models_payload(REQUIRED_MODEL_ALIAS),
        observe_fn=fake_observe,
    )
    formatted = report.format()
    assert "observed pid: 11" in formatted
    assert "model sha256 (observed): " + ("a" * 64) in formatted
    assert "llama-server --version (observed):" in formatted
