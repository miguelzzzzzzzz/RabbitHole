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
    WEIGHTS_MMAP_MIB,
    WEIGHTS_REPACKED_MIB,
    CheckResult,
    DoctorReport,
    ServeFlags,
    _check_serve_flags,
    base_url_from_environ,
    fetch_models,
    headroom_from_environ,
    intended_serve_flags,
    read_meminfo,
    required_ram_mib,
    run_doctor,
)

ROOT = Path(__file__).resolve().parents[1]
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

    report = run_doctor(environ=_pass_env(), meminfo_path=path, fetch_models_fn=boom)
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
    )
    assert not report.ok
    assert next(c for c in report.checks if c.name == "models").reason == "bad_models_response"


def test_doctor_meminfo_unreadable(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    report = run_doctor(
        environ=_pass_env(),
        meminfo_path=missing,
        fetch_models_fn=lambda _url: _models_payload(REQUIRED_MODEL_ALIAS),
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
    )
    result = _check_serve_flags(flags)
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
    assert "--jinja" not in text.split("serve flags:", 1)[1].splitlines()[0]
    assert "refuse" in text


def test_doctor_skips_models_when_requested(tmp_path: Path) -> None:
    need_kib = int(required_ram_mib(DEFAULT_MIN_FREE_MB) * 1024) + 10_000
    path = _write_meminfo(tmp_path / "meminfo", available_kib=need_kib)
    report = run_doctor(environ=_pass_env(), meminfo_path=path, skip_models=True)
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

    report = run_doctor(environ=_pass_env(), meminfo_path=path, fetch_models_fn=payload)
    assert report.ok


def test_doctor_format_includes_mem_when_unreadable_path_not_used(
    tmp_path: Path,
) -> None:
    # Cover the meminfo_unreadable path through format() as well.
    report = run_doctor(
        environ=_pass_env(),
        meminfo_path=tmp_path / "missing",
        fetch_models_fn=lambda _url: _models_payload(REQUIRED_MODEL_ALIAS),
    )
    text = report.format()
    assert "MemAvailable:" not in text
    assert "meminfo_unreadable" not in text  # reason is not printed; message is
    assert "cannot read meminfo" in text
