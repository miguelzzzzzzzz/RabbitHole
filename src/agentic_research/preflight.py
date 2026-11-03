"""Memory and local-server preflight for live runs (`arp doctor`).

Before any live local run the doctor checks, in order:

1. **Intended serve flags** match the pinned runtime: context length 4096 and
   the RAM prompt cache disabled (``--cache-ram 0``). The values come from the
   same source of truth as ``scripts/local_runtime.sh`` and
   ``docs/local-models.md`` (environment defaults + pinned constants), not from
   scraping a live process command line.
2. **Memory**: ``MemAvailable`` from ``/proc/meminfo`` covers the server's
   reported model + KV + compute buffers (about 1.8 GiB on the pinned
   runtime; more than GGUF + KV because weights are repacked) plus the
   ``ARP_MIN_FREE_MB`` headroom (default 1024). Swap is reported but never
   counted toward the budget; if ``MemAvailable`` is short the check refuses
   even when swap is present.
3. **Server**: ``GET {ARP_LLM_BASE_URL}/models`` answers and lists the pinned
   model id.

CI never starts the server. Unit tests inject a fake meminfo path and a fake
``/v1/models`` client so the fast suite stays offline.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

# --- Pins (must match scripts/local_runtime.sh and docs/local-models.md) -----

#: Context length the server must be started with (`-c` / ARP_LOCAL_CTX).
REQUIRED_CTX: Final = 4096
#: RAM prompt-cache limit in MiB (`--cache-ram`). Must be 0: this llama.cpp
#: build defaults to 8192 MiB, which the box cannot spare.
REQUIRED_CACHE_RAM_MIB: Final = 0
#: Server slots (`-np`); one slot => one KV cache.
REQUIRED_NP: Final = 1
#: Loopback host and port the script binds.
REQUIRED_HOST: Final = "127.0.0.1"
REQUIRED_PORT: Final = 8080
#: Served model id (`--alias` / ARP_LOCAL_MODEL).
REQUIRED_MODEL_ALIAS: Final = "Qwen3-1.7B-Q4_K_M"
#: Default OpenAI-compatible base URL (ARP_LLM_BASE_URL).
DEFAULT_BASE_URL: Final = f"http://{REQUIRED_HOST}:{REQUIRED_PORT}/v1"
#: Default preflight headroom on top of the reported buffers (ARP_MIN_FREE_MB).
DEFAULT_MIN_FREE_MB: Final = 1024

# Buffers llama-server reported when loading with the pinned flags
# (docs/local-models.md, 2026-11-02). Budget these, not the GGUF size alone.
WEIGHTS_REPACKED_MIB: Final = 1067.90
WEIGHTS_MMAP_MIB: Final = 243.44
KV_CACHE_MIB: Final = 448.00
COMPUTE_BUFFER_MIB: Final = 60.02
REPORTED_SERVER_BUFFERS_MIB: Final = (
    WEIGHTS_REPACKED_MIB + WEIGHTS_MMAP_MIB + KV_CACHE_MIB + COMPUTE_BUFFER_MIB
)  # ~1819.36


ModelsFetcher = Callable[[str], dict[str, Any]]


@dataclass(frozen=True)
class MemInfo:
    """Parsed fields from ``/proc/meminfo`` (Linux kB = KiB)."""

    mem_available_kib: int
    swap_total_kib: int

    @property
    def mem_available_mib(self) -> float:
        return self.mem_available_kib / 1024.0

    @property
    def swap_total_mib(self) -> float:
        return self.swap_total_kib / 1024.0


@dataclass(frozen=True)
class ServeFlags:
    """Intended llama-server launch flags for the pinned local runtime."""

    ctx: int
    cache_ram_mib: int
    np: int
    host: str
    port: int
    alias: str
    jinja: bool
    seed: int

    @property
    def ok(self) -> bool:
        return (
            self.ctx == REQUIRED_CTX
            and self.cache_ram_mib == REQUIRED_CACHE_RAM_MIB
            and self.np == REQUIRED_NP
            and self.host == REQUIRED_HOST
            and self.port == REQUIRED_PORT
            and self.alias == REQUIRED_MODEL_ALIAS
            and self.jinja is True
            and self.seed == 0
        )


@dataclass(frozen=True)
class CheckResult:
    """One named doctor check."""

    name: str
    ok: bool
    reason: str
    message: str
    details: Mapping[str, Any]


@dataclass(frozen=True)
class DoctorReport:
    """Aggregate result of ``run_doctor``."""

    ok: bool
    checks: tuple[CheckResult, ...]
    warnings: tuple[str, ...]
    required_mib: float
    headroom_mib: int
    reported_buffers_mib: float
    mem_available_mib: float | None
    swap_total_mib: float | None
    base_url: str
    model_id: str
    serve_flags: ServeFlags

    def format(self) -> str:
        lines = [
            "arp doctor",
            (
                f"required RAM: {self.reported_buffers_mib:.2f} MiB reported buffers "
                f"+ {self.headroom_mib} MiB headroom = {self.required_mib:.2f} MiB"
            ),
            (
                f"serve flags: -c {self.serve_flags.ctx} --cache-ram "
                f"{self.serve_flags.cache_ram_mib} -np {self.serve_flags.np} "
                f"--host {self.serve_flags.host} --port {self.serve_flags.port} "
                f"--alias {self.serve_flags.alias}"
                + (" --jinja" if self.serve_flags.jinja else "")
                + f" --seed {self.serve_flags.seed}"
            ),
            f"base URL: {self.base_url}",
        ]
        if self.mem_available_mib is not None:
            lines.append(
                f"MemAvailable: {self.mem_available_mib:.2f} MiB; "
                f"SwapTotal: {self.swap_total_mib:.2f} MiB"
            )
        for check in self.checks:
            mark = "ok" if check.ok else "FAIL"
            lines.append(f"[{mark}] {check.name}: {check.message}")
        for warning in self.warnings:
            lines.append(f"[warn] {warning}")
        lines.append("pass" if self.ok else "refuse")
        return "\n".join(lines)


def intended_serve_flags(environ: Mapping[str, str] | None = None) -> ServeFlags:
    """Return the serve flags the pinned runtime is configured to use.

    ``ARP_LOCAL_CTX`` and ``ARP_LOCAL_MODEL`` may override the defaults; the
    other flags are fixed to the values ``scripts/local_runtime.sh serve``
    passes (``--cache-ram 0``, ``-np 1``, loopback, ``--jinja``, ``--seed 0``).
    """
    env = os.environ if environ is None else environ
    ctx_raw = env.get("ARP_LOCAL_CTX", str(REQUIRED_CTX)).strip() or str(REQUIRED_CTX)
    alias = env.get("ARP_LOCAL_MODEL", REQUIRED_MODEL_ALIAS).strip() or REQUIRED_MODEL_ALIAS
    try:
        ctx = int(ctx_raw)
    except ValueError as exc:
        raise ValueError(f"ARP_LOCAL_CTX must be an integer (got {ctx_raw!r})") from exc
    return ServeFlags(
        ctx=ctx,
        cache_ram_mib=REQUIRED_CACHE_RAM_MIB,
        np=REQUIRED_NP,
        host=REQUIRED_HOST,
        port=REQUIRED_PORT,
        alias=alias,
        jinja=True,
        seed=0,
    )


def read_meminfo(path: Path | str = "/proc/meminfo") -> MemInfo:
    """Parse ``MemAvailable`` and ``SwapTotal`` from a meminfo file."""
    text = Path(path).read_text(encoding="utf-8")
    values: dict[str, int] = {}
    for line in text.splitlines():
        if not line or ":" not in line:
            continue
        key, rest = line.split(":", 1)
        parts = rest.split()
        if not parts:
            continue
        try:
            values[key] = int(parts[0])
        except ValueError:
            continue
    missing = [name for name in ("MemAvailable", "SwapTotal") if name not in values]
    if missing:
        raise ValueError(f"meminfo at {path} missing {', '.join(missing)}")
    return MemInfo(mem_available_kib=values["MemAvailable"], swap_total_kib=values["SwapTotal"])


def required_ram_mib(headroom_mib: int = DEFAULT_MIN_FREE_MB) -> float:
    """Reported server buffers plus headroom, in MiB."""
    if headroom_mib < 0:
        raise ValueError(f"headroom_mib must be >= 0 (got {headroom_mib})")
    return REPORTED_SERVER_BUFFERS_MIB + float(headroom_mib)


def headroom_from_environ(environ: Mapping[str, str] | None = None) -> int:
    env = os.environ if environ is None else environ
    raw = env.get("ARP_MIN_FREE_MB", str(DEFAULT_MIN_FREE_MB)).strip() or str(DEFAULT_MIN_FREE_MB)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"ARP_MIN_FREE_MB must be an integer (got {raw!r})") from exc
    if value < 0:
        raise ValueError(f"ARP_MIN_FREE_MB must be >= 0 (got {value})")
    return value


def base_url_from_environ(environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    return (env.get("ARP_LLM_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL).rstrip("/")


def fetch_models(base_url: str, *, timeout: float = 5.0) -> dict[str, Any]:
    """GET ``{base_url}/models`` and return the decoded JSON object."""
    url = f"{base_url.rstrip('/')}/models"
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise OSError(f"HTTP {exc.code} from {url}") from exc
    except urllib.error.URLError as exc:
        raise OSError(f"cannot reach {url}: {exc.reason}") from exc
    try:
        payload: Any = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OSError(f"invalid JSON from {url}") from exc
    if not isinstance(payload, dict):
        raise OSError(f"expected a JSON object from {url}, got {type(payload).__name__}")
    return payload


def _check_serve_flags(flags: ServeFlags) -> CheckResult:
    problems: list[str] = []
    if flags.ctx != REQUIRED_CTX:
        problems.append(f"context is {flags.ctx}, required {REQUIRED_CTX}")
    if flags.cache_ram_mib != REQUIRED_CACHE_RAM_MIB:
        problems.append(
            f"--cache-ram is {flags.cache_ram_mib}, required {REQUIRED_CACHE_RAM_MIB} "
            "(prompt cache must be disabled)"
        )
    if flags.np != REQUIRED_NP:
        problems.append(f"-np is {flags.np}, required {REQUIRED_NP}")
    if flags.host != REQUIRED_HOST or flags.port != REQUIRED_PORT:
        problems.append(
            f"bind is {flags.host}:{flags.port}, required {REQUIRED_HOST}:{REQUIRED_PORT}"
        )
    if flags.alias != REQUIRED_MODEL_ALIAS:
        problems.append(f"alias is {flags.alias!r}, required {REQUIRED_MODEL_ALIAS!r}")
    if not flags.jinja:
        problems.append("--jinja is required")
    if flags.seed != 0:
        problems.append(f"--seed is {flags.seed}, required 0")
    if problems:
        return CheckResult(
            name="serve_flags",
            ok=False,
            reason="bad_serve_flags",
            message="; ".join(problems),
            details={
                "ctx": flags.ctx,
                "cache_ram_mib": flags.cache_ram_mib,
                "np": flags.np,
                "host": flags.host,
                "port": flags.port,
                "alias": flags.alias,
                "jinja": flags.jinja,
                "seed": flags.seed,
            },
        )
    return CheckResult(
        name="serve_flags",
        ok=True,
        reason="ok",
        message=(
            f"context {flags.ctx}, --cache-ram {flags.cache_ram_mib}, "
            f"-np {flags.np}, prompt cache disabled"
        ),
        details={
            "ctx": flags.ctx,
            "cache_ram_mib": flags.cache_ram_mib,
            "np": flags.np,
            "host": flags.host,
            "port": flags.port,
            "alias": flags.alias,
            "jinja": flags.jinja,
            "seed": flags.seed,
        },
    )


def _check_memory(mem: MemInfo, *, required_mib: float, headroom_mib: int) -> CheckResult:
    available = mem.mem_available_mib
    details: dict[str, Any] = {
        "mem_available_mib": round(available, 2),
        "mem_available_kib": mem.mem_available_kib,
        "swap_total_mib": round(mem.swap_total_mib, 2),
        "swap_total_kib": mem.swap_total_kib,
        "required_mib": round(required_mib, 2),
        "reported_buffers_mib": round(REPORTED_SERVER_BUFFERS_MIB, 2),
        "headroom_mib": headroom_mib,
        "buffers": {
            "weights_repacked_mib": WEIGHTS_REPACKED_MIB,
            "weights_mmap_mib": WEIGHTS_MMAP_MIB,
            "kv_cache_mib": KV_CACHE_MIB,
            "compute_buffer_mib": COMPUTE_BUFFER_MIB,
        },
    }
    if available + 1e-9 < required_mib:
        shortfall = required_mib - available
        return CheckResult(
            name="memory",
            ok=False,
            reason="insufficient_memory",
            message=(
                f"MemAvailable {available:.2f} MiB < required {required_mib:.2f} MiB "
                f"(short by {shortfall:.2f} MiB; swap is not counted)"
            ),
            details=details,
        )
    return CheckResult(
        name="memory",
        ok=True,
        reason="ok",
        message=(
            f"MemAvailable {available:.2f} MiB >= required {required_mib:.2f} MiB "
            f"(buffers {REPORTED_SERVER_BUFFERS_MIB:.2f} + headroom {headroom_mib})"
        ),
        details=details,
    )


def _check_models(payload: dict[str, Any], *, expected_id: str, base_url: str) -> CheckResult:
    data = payload.get("data")
    if not isinstance(data, list):
        return CheckResult(
            name="models",
            ok=False,
            reason="bad_models_response",
            message=f"{base_url}/models did not return a data list",
            details={"payload_keys": sorted(payload)},
        )
    ids: list[str] = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            ids.append(item["id"])
    if expected_id not in ids:
        return CheckResult(
            name="models",
            ok=False,
            reason="model_missing",
            message=(f"{base_url}/models has ids {ids or '[]'}, expected {expected_id!r}"),
            details={"ids": ids, "expected_id": expected_id},
        )
    return CheckResult(
        name="models",
        ok=True,
        reason="ok",
        message=f"{base_url}/models lists {expected_id}",
        details={"ids": ids, "expected_id": expected_id},
    )


def run_doctor(
    *,
    environ: Mapping[str, str] | None = None,
    meminfo_path: Path | str = "/proc/meminfo",
    fetch_models_fn: ModelsFetcher | None = None,
    skip_models: bool = False,
) -> DoctorReport:
    """Run every preflight check and return a structured report.

    ``skip_models`` is for callers that only want flag + memory checks (for
    example a unit test that already covered the HTTP path separately). The
    CLI never sets it.
    """
    env = os.environ if environ is None else environ
    flags = intended_serve_flags(env)
    headroom = headroom_from_environ(env)
    required = required_ram_mib(headroom)
    base_url = base_url_from_environ(env)
    model_id = flags.alias

    checks: list[CheckResult] = [_check_serve_flags(flags)]
    warnings: list[str] = []

    mem_available_mib: float | None = None
    swap_total_mib: float | None = None
    try:
        mem = read_meminfo(meminfo_path)
    except (OSError, ValueError) as exc:
        checks.append(
            CheckResult(
                name="memory",
                ok=False,
                reason="meminfo_unreadable",
                message=f"cannot read meminfo at {meminfo_path}: {exc}",
                details={"path": str(meminfo_path)},
            )
        )
    else:
        mem_available_mib = mem.mem_available_mib
        swap_total_mib = mem.swap_total_mib
        checks.append(_check_memory(mem, required_mib=required, headroom_mib=headroom))
        if mem.swap_total_kib > 0:
            warnings.append(
                f"SwapTotal is {mem.swap_total_mib:.2f} MiB; preflight ignores swap "
                "and requires MemAvailable alone"
            )

    if skip_models:
        checks.append(
            CheckResult(
                name="models",
                ok=True,
                reason="skipped",
                message="models check skipped",
                details={},
            )
        )
    else:
        getter = fetch_models if fetch_models_fn is None else fetch_models_fn
        try:
            payload = getter(base_url)
        except OSError as exc:
            checks.append(
                CheckResult(
                    name="models",
                    ok=False,
                    reason="local_server_unavailable",
                    message=f"cannot reach {base_url}/models: {exc}",
                    details={"base_url": base_url},
                )
            )
        else:
            checks.append(_check_models(payload, expected_id=model_id, base_url=base_url))

    return DoctorReport(
        ok=all(check.ok for check in checks),
        checks=tuple(checks),
        warnings=tuple(warnings),
        required_mib=required,
        headroom_mib=headroom,
        reported_buffers_mib=REPORTED_SERVER_BUFFERS_MIB,
        mem_available_mib=mem_available_mib,
        swap_total_mib=swap_total_mib,
        base_url=base_url,
        model_id=model_id,
        serve_flags=flags,
    )
