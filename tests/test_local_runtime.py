"""Keep the local-runtime pins in sync and check the script's offline paths.

`scripts/local_runtime.sh` holds the pins it downloads and verifies;
`docs/local-models.md` documents them and `.env.example` points the client at
the server the script starts. Nothing here downloads, starts a server, or
calls a model.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

import pytest

from test_env_example import parse_env_example

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "local_runtime.sh"
DOC = ROOT / "docs" / "local-models.md"
ENV_EXAMPLE = ROOT / ".env.example"

_PIN = re.compile(r'^(?P<name>[A-Z][A-Z0-9_]*)="(?P<value>[^"$]*)"$', re.MULTILINE)
_SHA256 = re.compile(r"[0-9a-f]{64}")

needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


@pytest.fixture(scope="module")
def pins() -> dict[str, str]:
    """Literal NAME="value" assignments in the script (no expansions)."""
    return {m["name"]: m["value"] for m in _PIN.finditer(SCRIPT.read_text("utf-8"))}


@pytest.fixture(scope="module")
def doc() -> str:
    return DOC.read_text("utf-8")


@pytest.fixture(scope="module")
def env() -> dict[str, str]:
    return parse_env_example(ENV_EXAMPLE.read_text("utf-8"))


def test_script_has_all_pins(pins: dict[str, str]) -> None:
    expected = {
        "LLAMA_BUILD",
        "LLAMA_COMMIT",
        "LLAMA_SHA256",
        "GGUF_REPO",
        "GGUF_REVISION",
        "GGUF_FILE",
        "GGUF_SHA256",
        "MODEL_ALIAS",
        "HOST",
        "PORT",
    }
    assert sorted(expected - pins.keys()) == []


@pytest.mark.parametrize("name", ["LLAMA_SHA256", "GGUF_SHA256"])
def test_sha256_pins_are_full_digests(pins: dict[str, str], name: str) -> None:
    assert _SHA256.fullmatch(pins[name])


def test_gguf_revision_is_a_full_commit(pins: dict[str, str]) -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", pins["GGUF_REVISION"])


@pytest.mark.parametrize(
    "name",
    [
        "LLAMA_BUILD",
        "LLAMA_COMMIT",
        "LLAMA_SHA256",
        "GGUF_REPO",
        "GGUF_REVISION",
        "GGUF_FILE",
        "GGUF_SHA256",
        "MODEL_ALIAS",
    ],
)
def test_doc_records_each_pin(pins: dict[str, str], doc: str, name: str) -> None:
    assert pins[name] in doc


def test_doc_sha256_values_all_come_from_the_script(pins: dict[str, str], doc: str) -> None:
    # A digest in the doc that the script does not pin means the two drifted.
    assert set(_SHA256.findall(doc)) == {pins["LLAMA_SHA256"], pins["GGUF_SHA256"]}


def test_quantization_matches_file(pins: dict[str, str], doc: str) -> None:
    assert "Q4_K_M" in pins["GGUF_FILE"]
    assert "| Quantization | Q4_K_M |" in doc


def test_env_example_points_at_the_served_model(pins: dict[str, str], env: dict[str, str]) -> None:
    assert env["ARP_LLM_BASE_URL"] == f"http://{pins['HOST']}:{pins['PORT']}/v1"
    assert env["ARP_LOCAL_MODEL"] == pins["MODEL_ALIAS"]
    assert pins["HOST"] == "127.0.0.1"


def test_context_default_matches_env_example(env: dict[str, str]) -> None:
    script = SCRIPT.read_text("utf-8")
    assert f'CTX="${{ARP_LOCAL_CTX:-{env["ARP_LOCAL_CTX"]}}}"' in script


@pytest.mark.parametrize(
    "flag",
    [
        "--jinja",
        "--cache-ram 0",
        "-np 1",
        '--host "$HOST"',
        '--port "$PORT"',
        '-c "$CTX"',
        "--reasoning-format none",
    ],
)
def test_serve_passes_memory_and_template_flags(flag: str) -> None:
    serve = SCRIPT.read_text("utf-8").split("serve() {", 1)[1].split("\n}\n", 1)[0]
    assert flag in serve


@needs_bash
def test_script_parses() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


@needs_bash
def test_unknown_command_prints_usage() -> None:
    proc = subprocess.run(["bash", str(SCRIPT), "nope"], capture_output=True, text=True)
    assert proc.returncode == 2
    assert "usage:" in proc.stderr


@needs_bash
@pytest.mark.parametrize("command", ["verify", "serve"])
def test_refuses_to_run_without_verified_files(tmp_path: Path, command: str) -> None:
    proc = subprocess.run(
        ["bash", str(SCRIPT), command],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "ARP_RUNTIME_DIR": str(tmp_path)},
    )
    assert proc.returncode == 1
    assert "missing or bad" in proc.stderr
    assert "run: " in proc.stderr


@needs_bash
def test_verify_rejects_a_file_with_the_wrong_digest(tmp_path: Path, pins: dict[str, str]) -> None:
    asset = f"llama-{pins['LLAMA_BUILD']}-bin-ubuntu-x64.tar.gz"
    (tmp_path / asset).write_bytes(b"not the release asset")
    proc = subprocess.run(
        ["bash", str(SCRIPT), "verify"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "ARP_RUNTIME_DIR": str(tmp_path)},
    )
    assert proc.returncode == 1
    assert f"missing or bad {tmp_path / asset}" in proc.stderr


@pytest.mark.slow
@pytest.mark.allow_network
def test_live_server_serves_the_pinned_model(env: dict[str, str]) -> None:
    """Needs `scripts/local_runtime.sh serve` running; never runs in CI."""
    base_url = os.environ.get("ARP_LLM_BASE_URL") or env["ARP_LLM_BASE_URL"]
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=10) as resp:
            payload = json.load(resp)
    except OSError as exc:
        pytest.fail(f"no server at {base_url} ({exc}); run scripts/local_runtime.sh serve")
    ids = [model["id"] for model in payload["data"]]
    assert ids == [env["ARP_LOCAL_MODEL"]]
