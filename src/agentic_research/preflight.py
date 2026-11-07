"""Memory and local-server preflight for live runs (`arp doctor`).

Before any live local run the doctor checks, in order:

1. **Observed serve flags** from the live ``llama-server`` process listening on
   ``ARP_LLM_BASE_URL``'s port: read ``/proc/<pid>/cmdline`` and cross-check
   ``GET /props`` when available. Refuse if the process cannot be observed.
   Required: ``-c 4096``, ``--cache-ram 0``, ``-np 1``, ``--reasoning-format
   none``, loopback bind, ``--jinja``, ``--seed 0``, alias
   ``Qwen3-1.7B-Q4_K_M``. Pins match ``scripts/local_runtime.sh`` /
   ``docs/local-models.md``.
2. **Memory**: ``MemAvailable`` from ``/proc/meminfo`` covers the server's
   reported model + KV + compute buffers (about 1.8 GiB on the pinned
   runtime; more than GGUF + KV because weights are repacked) plus the
   ``ARP_MIN_FREE_MB`` headroom (default 1024). Swap is reported but never
   counted toward the budget; if ``MemAvailable`` is short the check refuses
   even when swap is present.
3. **Server**: ``GET {ARP_LLM_BASE_URL}/models`` answers and lists the pinned
   model id.

CI never starts the server. Unit tests inject a fake meminfo path, a fake
``/v1/models`` client, and a fake observe callback so the fast suite stays
offline.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
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
#: Reasoning format: keep <think> in content for probe scoring.
REQUIRED_REASONING_FORMAT: Final = "none"
#: Default OpenAI-compatible base URL (ARP_LLM_BASE_URL).
DEFAULT_BASE_URL: Final = f"http://{REQUIRED_HOST}:{REQUIRED_PORT}/v1"
#: Default preflight headroom on top of the reported buffers (ARP_MIN_FREE_MB).
DEFAULT_MIN_FREE_MB: Final = 1024
#: Pinned GGUF sha256 (docs/local-models.md); used when observation fails to
#: hash the served path and for provenance labels.
PINNED_GGUF_SHA256: Final = "b139949c5bd74937ad8ed8c8cf3d9ffb1e99c866c823204dc42c0d91fa181897"
PINNED_LLAMA_BUILD: Final = "b12032"
PINNED_LLAMA_VERSION: Final = "0.8.0-dev (build 12032, commit dde879bc4)"

# Buffers llama-server reported when loading with the pinned flags
# (docs/local-models.md, 2026-11-02). Budget these, not the GGUF size alone.
WEIGHTS_REPACKED_MIB: Final = 1067.90
WEIGHTS_MMAP_MIB: Final = 243.44
KV_CACHE_MIB: Final = 448.00
COMPUTE_BUFFER_MIB: Final = 60.02
REPORTED_SERVER_BUFFERS_MIB: Final = (
    WEIGHTS_REPACKED_MIB + WEIGHTS_MMAP_MIB + KV_CACHE_MIB + COMPUTE_BUFFER_MIB
)  # ~1819.36

_TCP_LISTEN: Final = 0x0A

ModelsFetcher = Callable[[str], dict[str, Any]]
PropsFetcher = Callable[[str], dict[str, Any] | None]


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
    """llama-server launch flags (intended pins or observed live values)."""

    ctx: int
    cache_ram_mib: int
    np: int
    host: str
    port: int
    alias: str
    jinja: bool
    seed: int
    reasoning_format: str
    model_path: str | None = None

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
            and self.reasoning_format == REQUIRED_REASONING_FORMAT
        )

    def launch_flags_summary(self) -> str:
        model = self.model_path if self.model_path else "<GGUF>"
        return (
            f"llama-server -m {model} --alias {self.alias} "
            f"--host {self.host} --port {self.port} -c {self.ctx} "
            f"-np {self.np}"
            + (" --jinja" if self.jinja else " --no-jinja")
            + f" --cache-ram {self.cache_ram_mib} --seed {self.seed} "
            f"--reasoning-format {self.reasoning_format}"
        )


@dataclass(frozen=True)
class ObservedRuntime:
    """Live observation of the process serving ``ARP_LLM_BASE_URL``."""

    pid: int
    cmdline: tuple[str, ...]
    flags: ServeFlags
    props: dict[str, Any] | None
    model_sha256: str | None
    llama_version: str | None
    binary_path: str | None
    observation_error: str | None = None

    @property
    def ok(self) -> bool:
        return self.observation_error is None and self.flags.ok


ObserveFn = Callable[..., ObservedRuntime]


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
    observed: ObservedRuntime | None = None

    def format(self) -> str:
        lines = [
            "arp doctor",
            (
                f"required RAM: {self.reported_buffers_mib:.2f} MiB reported buffers "
                f"+ {self.headroom_mib} MiB headroom = {self.required_mib:.2f} MiB"
            ),
            (
                f"serve flags (observed): -c {self.serve_flags.ctx} --cache-ram "
                f"{self.serve_flags.cache_ram_mib} -np {self.serve_flags.np} "
                f"--host {self.serve_flags.host} --port {self.serve_flags.port} "
                f"--alias {self.serve_flags.alias}"
                + (" --jinja" if self.serve_flags.jinja else "")
                + f" --seed {self.serve_flags.seed}"
                + f" --reasoning-format {self.serve_flags.reasoning_format}"
            ),
            f"base URL: {self.base_url}",
        ]
        if self.observed is not None:
            lines.append(f"observed pid: {self.observed.pid}")
            if self.observed.model_sha256:
                lines.append(f"model sha256 (observed): {self.observed.model_sha256}")
            if self.observed.llama_version:
                lines.append(f"llama-server --version (observed): {self.observed.llama_version}")
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
    passes (``--cache-ram 0``, ``-np 1``, loopback, ``--jinja``, ``--seed 0``,
    ``--reasoning-format none``). Doctor checks the *live* process, not this
    helper; tests and docs use it as the pin source of truth.
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
        reasoning_format=REQUIRED_REASONING_FORMAT,
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


def port_from_base_url(base_url: str) -> int:
    """Extract the TCP port from an OpenAI-compatible base URL."""
    parsed = urllib.parse.urlparse(base_url if "://" in base_url else f"http://{base_url}")
    if parsed.port is not None:
        return parsed.port
    if parsed.scheme == "https":
        return 443
    return 80


def server_root_from_base_url(base_url: str) -> str:
    """Strip a trailing ``/v1`` to get the llama-server root (for ``/props``)."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return root or base_url


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


def fetch_props(base_url: str, *, timeout: float = 5.0) -> dict[str, Any] | None:
    """GET ``{server_root}/props``; return None if unreachable or not JSON."""
    url = f"{server_root_from_base_url(base_url)}/props"
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        return None
    try:
        payload: Any = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _listening_socket_inodes(port: int, *, proc_root: Path) -> set[int]:
    """Return socket inodes in LISTEN state for ``port`` from ``/proc/net/tcp{,6}``."""
    inodes: set[int] = set()
    for name in ("tcp", "tcp6"):
        path = proc_root / "net" / name
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 10:
                continue
            local = parts[1]
            state_hex = parts[3]
            inode_s = parts[9]
            try:
                state = int(state_hex, 16)
                inode = int(inode_s)
            except ValueError:
                continue
            if state != _TCP_LISTEN or inode <= 0:
                continue
            if ":" not in local:
                continue
            _addr, port_hex = local.rsplit(":", 1)
            try:
                local_port = int(port_hex, 16)
            except ValueError:
                continue
            if local_port == port:
                inodes.add(inode)
    return inodes


def _pids_holding_inodes(inodes: set[int], *, proc_root: Path) -> list[int]:
    if not inodes:
        return []
    found: list[int] = []
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return []
    for entry in entries:
        name = entry.name
        if not name.isdigit():
            continue
        fd_dir = entry / "fd"
        try:
            fds = list(fd_dir.iterdir())
        except OSError:
            continue
        for fd in fds:
            try:
                target = str(fd.readlink())
            except OSError:
                continue
            if not target.startswith("socket:["):
                continue
            try:
                inode = int(target[len("socket:[") : -1])
            except ValueError:
                continue
            if inode in inodes:
                found.append(int(name))
                break
    return sorted(set(found))


def find_listener_pid(port: int, *, proc_root: Path | str = "/proc") -> int | None:
    """Return the PID listening on ``port``, or None if not found."""
    root = Path(proc_root)
    inodes = _listening_socket_inodes(port, proc_root=root)
    pids = _pids_holding_inodes(inodes, proc_root=root)
    if not pids:
        return None
    return pids[0]


def read_cmdline(pid: int, *, proc_root: Path | str = "/proc") -> list[str]:
    """Read ``/proc/<pid>/cmdline`` as a list of arguments."""
    raw = (Path(proc_root) / str(pid) / "cmdline").read_bytes()
    if not raw:
        raise OSError(f"empty cmdline for pid {pid}")
    parts = [p.decode("utf-8", "surrogateescape") for p in raw.split(b"\0") if p]
    if not parts:
        raise OSError(f"empty cmdline for pid {pid}")
    return parts


def _flag_value(argv: Sequence[str], *names: str) -> str | None:
    """Return the value for the first matching ``--flag value`` / ``--flag=value``."""
    name_set = set(names)
    i = 0
    while i < len(argv):
        arg = argv[i]
        for name in name_set:
            if arg == name:
                if i + 1 < len(argv):
                    return argv[i + 1]
                return None
            prefix = name + "="
            if arg.startswith(prefix):
                return arg[len(prefix) :]
        i += 1
    return None


def _has_flag(argv: Sequence[str], *names: str) -> bool:
    name_set = set(names)
    return any(arg in name_set or any(arg.startswith(n + "=") for n in name_set) for arg in argv)


def parse_cmdline_flags(argv: Sequence[str]) -> ServeFlags:
    """Parse llama-server argv into ``ServeFlags``.

    Missing ``--reasoning-format`` is treated as ``auto`` (the server default),
    which fails the required-``none`` check. Missing ``--cache-ram`` is treated
    as the build default 8192 so doctor refuses an unpinned cache.
    """
    ctx_raw = _flag_value(argv, "-c", "--ctx-size")
    cache_raw = _flag_value(argv, "--cache-ram")
    np_raw = _flag_value(argv, "-np", "--parallel")
    host = _flag_value(argv, "--host") or REQUIRED_HOST
    port_raw = _flag_value(argv, "--port")
    alias = _flag_value(argv, "--alias") or ""
    seed_raw = _flag_value(argv, "--seed")
    reasoning = _flag_value(argv, "--reasoning-format")
    model_path = _flag_value(argv, "-m", "--model")

    def _int(raw: str | None, default: int) -> int:
        if raw is None:
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    jinja = True
    if _has_flag(argv, "--no-jinja"):
        jinja = False
    elif _has_flag(argv, "--jinja"):
        jinja = True

    return ServeFlags(
        ctx=_int(ctx_raw, -1),
        cache_ram_mib=_int(cache_raw, 8192),  # server default when omitted
        np=_int(np_raw, -1),
        host=host,
        port=_int(port_raw, -1),
        alias=alias,
        jinja=jinja,
        seed=_int(seed_raw, -1),
        reasoning_format=reasoning if reasoning is not None else "auto",
        model_path=model_path,
    )


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def llama_server_version(binary: Path | str) -> str | None:
    """Run ``binary --version`` and return the ``version:`` line body, if any."""
    try:
        proc = subprocess.run(
            [str(binary), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = (proc.stdout or "") + (proc.stderr or "")
    for line in text.splitlines():
        if line.startswith("version:"):
            return line.split(":", 1)[1].strip()
    return None


def observe_runtime(
    *,
    base_url: str,
    proc_root: Path | str = "/proc",
    fetch_props_fn: PropsFetcher | None = None,
    pid: int | None = None,
    cmdline: Sequence[str] | None = None,
    hash_model: bool = True,
    read_version: bool = True,
) -> ObservedRuntime:
    """Observe the live llama-server for ``base_url``.

    Tests may inject ``pid`` + ``cmdline`` (and optional props) to avoid a
    real process. When ``cmdline`` is injected, ``pid`` defaults to 0.
    """
    props_getter = fetch_props if fetch_props_fn is None else fetch_props_fn
    props = props_getter(base_url)

    if cmdline is not None:
        argv = list(cmdline)
        observed_pid = 0 if pid is None else pid
    else:
        port = port_from_base_url(base_url)
        found = find_listener_pid(port, proc_root=proc_root) if pid is None else pid
        if found is None:
            return ObservedRuntime(
                pid=-1,
                cmdline=(),
                flags=ServeFlags(
                    ctx=-1,
                    cache_ram_mib=-1,
                    np=-1,
                    host="",
                    port=port,
                    alias="",
                    jinja=False,
                    seed=-1,
                    reasoning_format="unreadable",
                ),
                props=props,
                model_sha256=None,
                llama_version=None,
                binary_path=None,
                observation_error=(
                    f"no process listening on port {port} for base URL {base_url}; "
                    "cannot observe launch flags"
                ),
            )
        try:
            argv = read_cmdline(found, proc_root=proc_root)
        except OSError as exc:
            return ObservedRuntime(
                pid=found,
                cmdline=(),
                flags=ServeFlags(
                    ctx=-1,
                    cache_ram_mib=-1,
                    np=-1,
                    host="",
                    port=port,
                    alias="",
                    jinja=False,
                    seed=-1,
                    reasoning_format="unreadable",
                ),
                props=props,
                model_sha256=None,
                llama_version=None,
                binary_path=None,
                observation_error=f"cannot read /proc/{found}/cmdline: {exc}",
            )
        observed_pid = found

    flags = parse_cmdline_flags(argv)
    # Cross-check /props when present.
    if props is not None:
        model_path_props = props.get("model_path")
        if isinstance(model_path_props, str) and model_path_props and flags.model_path is None:
            flags = ServeFlags(
                ctx=flags.ctx,
                cache_ram_mib=flags.cache_ram_mib,
                np=flags.np,
                host=flags.host,
                port=flags.port,
                alias=flags.alias,
                jinja=flags.jinja,
                seed=flags.seed,
                reasoning_format=flags.reasoning_format,
                model_path=model_path_props,
            )
        settings = props.get("default_generation_settings")
        if isinstance(settings, dict):
            n_ctx = settings.get("n_ctx")
            if isinstance(n_ctx, int) and flags.ctx < 0:
                flags = ServeFlags(
                    ctx=n_ctx,
                    cache_ram_mib=flags.cache_ram_mib,
                    np=flags.np,
                    host=flags.host,
                    port=flags.port,
                    alias=flags.alias,
                    jinja=flags.jinja,
                    seed=flags.seed,
                    reasoning_format=flags.reasoning_format,
                    model_path=flags.model_path,
                )
            params = settings.get("params")
            if isinstance(params, dict):
                rf = params.get("reasoning_format")
                if isinstance(rf, str) and flags.reasoning_format == "auto":
                    # Only fill from props when cmdline omitted the flag.
                    pass  # cmdline already defaulted to auto; props may confirm
        alias_props = props.get("model_alias")
        if isinstance(alias_props, str) and alias_props and not flags.alias:
            flags = ServeFlags(
                ctx=flags.ctx,
                cache_ram_mib=flags.cache_ram_mib,
                np=flags.np,
                host=flags.host,
                port=flags.port,
                alias=alias_props,
                jinja=flags.jinja,
                seed=flags.seed,
                reasoning_format=flags.reasoning_format,
                model_path=flags.model_path,
            )
        total_slots = props.get("total_slots")
        if isinstance(total_slots, int) and flags.np < 0:
            flags = ServeFlags(
                ctx=flags.ctx,
                cache_ram_mib=flags.cache_ram_mib,
                np=total_slots,
                host=flags.host,
                port=flags.port,
                alias=flags.alias,
                jinja=flags.jinja,
                seed=flags.seed,
                reasoning_format=flags.reasoning_format,
                model_path=flags.model_path,
            )

    binary = argv[0] if argv else None
    model_sha: str | None = None
    if hash_model and flags.model_path:
        try:
            model_sha = sha256_file(flags.model_path)
        except OSError:
            model_sha = None
    version: str | None = None
    if read_version and binary:
        version = llama_server_version(binary)

    return ObservedRuntime(
        pid=observed_pid,
        cmdline=tuple(argv),
        flags=flags,
        props=props,
        model_sha256=model_sha,
        llama_version=version,
        binary_path=binary,
        observation_error=None,
    )


def _check_serve_flags(flags: ServeFlags, *, observed: bool) -> CheckResult:
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
    if flags.reasoning_format != REQUIRED_REASONING_FORMAT:
        problems.append(
            f"--reasoning-format is {flags.reasoning_format!r}, "
            f"required {REQUIRED_REASONING_FORMAT!r}"
        )
    source = "observed" if observed else "intended"
    if problems:
        return CheckResult(
            name="serve_flags",
            ok=False,
            reason="bad_serve_flags",
            message=f"({source}) " + "; ".join(problems),
            details={
                "source": source,
                "ctx": flags.ctx,
                "cache_ram_mib": flags.cache_ram_mib,
                "np": flags.np,
                "host": flags.host,
                "port": flags.port,
                "alias": flags.alias,
                "jinja": flags.jinja,
                "seed": flags.seed,
                "reasoning_format": flags.reasoning_format,
                "model_path": flags.model_path,
            },
        )
    return CheckResult(
        name="serve_flags",
        ok=True,
        reason="ok",
        message=(
            f"({source}) context {flags.ctx}, --cache-ram {flags.cache_ram_mib}, "
            f"-np {flags.np}, --reasoning-format {flags.reasoning_format}, "
            "prompt cache disabled"
        ),
        details={
            "source": source,
            "ctx": flags.ctx,
            "cache_ram_mib": flags.cache_ram_mib,
            "np": flags.np,
            "host": flags.host,
            "port": flags.port,
            "alias": flags.alias,
            "jinja": flags.jinja,
            "seed": flags.seed,
            "reasoning_format": flags.reasoning_format,
            "model_path": flags.model_path,
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
    observe_fn: ObserveFn | None = None,
    skip_models: bool = False,
    skip_observe: bool = False,
) -> DoctorReport:
    """Run every preflight check and return a structured report.

    ``skip_models`` / ``skip_observe`` are for unit tests. The CLI never sets
    them. When ``skip_observe`` is True, serve-flag checks use intended pins
    (legacy test path); production always observes the live process.
    """
    env = os.environ if environ is None else environ
    intended = intended_serve_flags(env)
    headroom = headroom_from_environ(env)
    required = required_ram_mib(headroom)
    base_url = base_url_from_environ(env)
    model_id = intended.alias

    checks: list[CheckResult] = []
    warnings: list[str] = []
    observed: ObservedRuntime | None = None
    flags_for_report = intended

    if skip_observe:
        checks.append(_check_serve_flags(intended, observed=False))
    else:
        observer = observe_runtime if observe_fn is None else observe_fn
        observed = observer(base_url=base_url)
        if observed.observation_error is not None:
            checks.append(
                CheckResult(
                    name="serve_flags",
                    ok=False,
                    reason="serve_flags_unreadable",
                    message=observed.observation_error,
                    details={"base_url": base_url, "pid": observed.pid},
                )
            )
            flags_for_report = observed.flags
        else:
            flags_for_report = observed.flags
            checks.append(_check_serve_flags(observed.flags, observed=True))

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
        serve_flags=flags_for_report,
        observed=observed,
    )
