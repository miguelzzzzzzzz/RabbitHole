"""``fetch_url`` (PLAN M3, ADR-0007): fixture pages by default, live HTTP only by opt-in.

The tool runs through the real registry and executor. Live fetching is
exercised with `httpx.MockTransport` (no socket); the socket tests replace the
autouse network guard with a recorder to prove that, without opt-in and an
allowlisted host, no connection or DNS lookup is even attempted.
"""

from __future__ import annotations

import hashlib
import json
import socket
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import httpx
import jsonschema
import pytest

from agentic_research.agent import AgentLoop, Budgets, FinalAnswer, GiveUp, Task, ToolCall
from agentic_research.agent.actions import Citation, Claim
from agentic_research.agent.evidence import EvidenceStore
from agentic_research.backends import FixtureSearchBackend
from agentic_research.citations import INVALID_CITATION, CitationVerifier
from agentic_research.policies.oracle import OraclePolicy
from agentic_research.tools import ToolContext, ToolExecution, ToolExecutor, ToolRegistry
from agentic_research.tools.builtin import catalog_registry, research_registry
from agentic_research.tools.fetch import (
    MAX_FETCH_CHARS,
    MAX_LIVE_BYTES,
    MAX_TITLE_CHARS,
    MAX_URL_CHARS,
    FetchUrlArgs,
    FetchUrlOutput,
    FetchUrlTool,
    LiveFetchPolicy,
    html_to_text,
    load_fixture_pages,
    normalize_url,
    plain_text,
)
from rag_index import FIXTURE_CORPUS, FIXTURE_CORPUS_ID

PAGES_DIR = Path(__file__).parent / "fixtures" / "pages"
VISITOR_URL = "https://visit-harrowvale.example.org/velmora-bridge"
TOLL_URL = "https://osk-river-trust.example.com/notices/2024-tolls"
LIVE_HOST = "docs.example.net"
LIVE_URL = f"https://{LIVE_HOST}/page"
CTX = ToolContext(run_id="r-fetch", step=0)


async def _no_sleep(_: float) -> None:
    return None


async def _execute(tool: FetchUrlTool, args: Mapping[str, Any] | str) -> ToolExecution:
    executor = ToolExecutor(ToolRegistry([tool]), sleeper=_no_sleep)
    return await executor.execute("fetch_url", args, CTX)


async def _ok(tool: FetchUrlTool, args: Mapping[str, Any]) -> FetchUrlOutput:
    execution = await _execute(tool, args)
    assert execution.result.error is None, execution.result.error
    return FetchUrlOutput.model_validate(execution.result.output)


async def _error(tool: FetchUrlTool, args: Mapping[str, Any] | str) -> tuple[str, str, str]:
    error = (await _execute(tool, args)).result.error
    assert error is not None
    return error.error_class.value, error.reason, error.message


def _fixture_tool(**kwargs: Any) -> FetchUrlTool:
    return FetchUrlTool(load_fixture_pages(PAGES_DIR), **kwargs)


class _Server:
    """A `MockTransport` handler that records requests and returns a canned response."""

    def __init__(self, respond: Callable[[httpx.Request], httpx.Response]) -> None:
        self.respond = respond
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.respond(request)

    def tool(
        self, *, allow: str = LIVE_HOST, pages: Mapping[str, str] | None = None
    ) -> FetchUrlTool:
        live = LiveFetchPolicy(enabled=True, allowlist=frozenset({allow}))
        return FetchUrlTool(pages, live=live, transport=httpx.MockTransport(self))


def _html(body: str, status: int = 200, content_type: str = "text/html; charset=utf-8") -> Any:
    def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=body.encode(), headers={"content-type": content_type})

    return respond


# --- URLs -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://example.org", "https://example.org/"),
        ("HTTPS://Example.ORG/Path?q=1#frag", "https://example.org/Path?q=1"),
        ("http://example.org:80/a", "http://example.org/a"),
        ("https://example.org:443/a", "https://example.org/a"),
        ("https://example.org:8443/a", "https://example.org:8443/a"),
        ("https://example.org./a", "https://example.org/a"),
        ("http://127.0.0.1:8000/x", "http://127.0.0.1:8000/x"),
        ("http://[::1]/x", "http://[::1]/x"),
        ("https://a-b.example.org/%E2%9C%93", "https://a-b.example.org/%E2%9C%93"),
    ],
)
def test_normalize_url(url: str, expected: str) -> None:
    assert normalize_url(url) == expected
    assert normalize_url(expected) == expected  # idempotent


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("ftp://example.org/a", "only http and https"),
        ("file:///etc/passwd", "only http and https"),
        ("javascript:alert(1)", "only http and https"),
        ("example.org/a", "only http and https"),
        ("https:///path", "no host"),
        ("https://user:pw@example.org/", "user info"),
        ("https://user@example.org/", "user info"),
        ("https://exa mple.org/", "whitespace or control"),
        ("https://example.org/\n", "whitespace or control"),
        ("https://exämple.org/", "ASCII"),
        ("https://example.org:99999/", "malformed URL"),
        ("https://[::1/", "malformed URL"),
        ("https://-bad-.example.org/", "invalid host"),
        ("https://exa_mple.org/", "invalid host"),
        ("https://[fe80::1%25eth0]/", "invalid host"),
        ("https://example.org/" + "a" * MAX_URL_CHARS, "longer than"),
    ],
)
def test_normalize_url_refuses(url: str, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        normalize_url(url)


def test_live_fetch_is_off_by_default() -> None:
    assert LiveFetchPolicy() == LiveFetchPolicy.from_env({})
    assert not LiveFetchPolicy().enabled
    # An allowlisted host is still refused while live fetch is off.
    assert not LiveFetchPolicy(enabled=False, allowlist=frozenset({LIVE_HOST})).allows(LIVE_HOST)


@pytest.mark.parametrize("flag", ["true", "TRUE", " 1 ", "yes", "on"])
def test_live_fetch_flag_on(flag: str) -> None:
    policy = LiveFetchPolicy.from_env(
        {"ARP_ALLOW_LIVE_FETCH": flag, "ARP_FETCH_ALLOWLIST": " Docs.Example.NET. ,, b.example "}
    )
    assert policy.enabled
    assert policy.allowlist == frozenset({LIVE_HOST, "b.example"})
    assert policy.allows(LIVE_HOST)
    # Exact host names: no subdomains, no parents.
    assert not policy.allows("www.docs.example.net")
    assert not policy.allows("example.net")


@pytest.mark.parametrize("flag", ["", "false", "0", "no", "off", " False "])
def test_live_fetch_flag_off(flag: str) -> None:
    assert not LiveFetchPolicy.from_env({"ARP_ALLOW_LIVE_FETCH": flag}).enabled


def test_live_fetch_config_errors_are_not_guessed() -> None:
    with pytest.raises(ValueError, match="ARP_ALLOW_LIVE_FETCH must be true or false"):
        LiveFetchPolicy.from_env({"ARP_ALLOW_LIVE_FETCH": "maybe"})
    for entry in ("*.example.org", "https://example.org", "exa mple.org", "a/b"):
        with pytest.raises(ValueError, match="is not a host name"):
            LiveFetchPolicy.from_env({"ARP_FETCH_ALLOWLIST": f"ok.example,{entry}"})


# --- HTML to text -----------------------------------------------------------------------


def test_html_to_text_drops_hidden_content_and_keeps_block_lines() -> None:
    page = html_to_text(
        "<html><head><title> A &amp;\n B </title><style>p{}</style>"
        "<script>evil()</script></head><body>"
        "<h1>Head</h1><p>one   two\n three</p><p>x<br>y<br/>z</p>"
        "<noscript>no</noscript><template><p>tpl</p></template>"
        "<svg><title>logo</title><text>svg</text></svg>"
        "<ul><li>a &lt; b</li><li>&#x2713;</li></ul><span>in</span><b>line</b></body></html>"
    )
    assert page.title == "A & B"
    assert page.text == "Head\none two three\nx\ny\nz\na < b\n\u2713\ninline"
    assert page.content_hash == hashlib.sha256(page.text.encode()).hexdigest()


def test_html_to_text_tolerates_broken_nesting() -> None:
    # An unclosed tag inside a skipped one is closed with it; a stray end tag is ignored.
    page = html_to_text("<noscript><svg>hidden</noscript>shown</svg> too <svg/>visible")
    assert page.text == "shown too visible"
    # An unclosed skipped element hides the rest of the page.
    assert html_to_text("<p>a</p><template>b<p>c").text == "a"


def test_html_to_text_title_is_capped_and_optional() -> None:
    assert html_to_text("<p>x</p>").title == ""
    assert html_to_text(f"<title>{'t' * 500}</title>").title == "t" * MAX_TITLE_CHARS
    assert html_to_text("<title>a</title>").text == ""


def test_plain_text_uses_the_same_line_rules() -> None:
    page = plain_text("  first   line \r\n\n\tsecond\n   \n")
    assert page == html_to_text("<p>first line</p><p>second</p>")


def test_fixture_pages_extract_as_documented() -> None:
    pages = load_fixture_pages(PAGES_DIR)
    assert set(pages) == {VISITOR_URL, TOLL_URL}
    visitor = html_to_text(pages[VISITOR_URL])
    assert visitor.title == "Velmora Bridge & Visitor Centre | Harrow Vale"
    assert "It opened in 1986, on the bridge's 75th anniversary." in visitor.text
    assert "April to October: 09:00\u201318:00" in visitor.text
    for hidden in ("tracking", "font-family", "Enable scripts", "Template text"):
        assert hidden not in visitor.text
    toll = html_to_text(pages[TOLL_URL])
    assert "Car\n4 crowns\nLorry\n11 crowns" in toll.text
    assert "logo" not in toll.text and toll.title == "Osk River Trust: toll notice"


# --- fixture loading --------------------------------------------------------------------


def _write_index(tmp_path: Path, index: object, files: Mapping[str, str] | None = None) -> Path:
    (tmp_path / "pages.json").write_text(json.dumps(index), encoding="utf-8")
    for name, text in (files or {}).items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    ("index", "message"),
    [
        ([], "must hold a 'pages' list"),
        ({"pages": {}}, "must hold a 'pages' list"),
        ({"pages": [{"url": VISITOR_URL}]}, "exactly 'url' and 'file'"),
        ({"pages": ["x"]}, "exactly 'url' and 'file'"),
        ({"pages": [{"url": 1, "file": "a.html"}]}, "must be strings"),
        ({"pages": [{"url": "ftp://x.example/", "file": "a.html"}]}, "only http and https"),
        ({"pages": [{"url": VISITOR_URL, "file": "missing.html"}]}, "is not a file under"),
        ({"pages": [{"url": VISITOR_URL, "file": "../outside.html"}]}, "is not a file under"),
        (
            {
                "pages": [
                    {"url": VISITOR_URL, "file": "a.html"},
                    {
                        "url": VISITOR_URL.upper().replace("VELMORA-BRIDGE", "velmora-bridge"),
                        "file": "a.html",
                    },
                ]
            },
            "duplicate fixture URL",
        ),
    ],
)
def test_load_fixture_pages_refuses_bad_indexes(
    tmp_path: Path, index: object, message: str
) -> None:
    (tmp_path.parent / "outside.html").write_text("<p>x</p>", encoding="utf-8")
    directory = _write_index(tmp_path, index, {"a.html": "<p>a</p>"})
    with pytest.raises(ValueError, match=message):
        load_fixture_pages(directory)


def test_tool_refuses_duplicate_and_invalid_fixture_urls() -> None:
    with pytest.raises(ValueError, match="duplicate fixture URL"):
        FetchUrlTool({"https://a.example/": "x", "HTTPS://A.EXAMPLE": "y"})
    with pytest.raises(ValueError, match="only http and https"):
        FetchUrlTool({"a.example/": "x"})


# --- the tool over fixture pages --------------------------------------------------------


def test_schema_is_openai_format() -> None:
    (schema,) = ToolRegistry([FetchUrlTool()]).openai_schemas()
    assert schema["function"]["name"] == "fetch_url"
    assert "untrusted" in schema["function"]["description"]
    params = schema["function"]["parameters"]
    jsonschema.Draft202012Validator.check_schema(params)
    assert params == FetchUrlArgs.model_json_schema()
    assert params["required"] == ["url"]
    assert params["additionalProperties"] is False
    assert params["properties"]["start"]["default"] == 0


async def test_fetches_a_fixture_page_with_offsets() -> None:
    tool = _fixture_tool()
    text = html_to_text(load_fixture_pages(PAGES_DIR)[VISITOR_URL]).text
    output = await _ok(tool, {"url": VISITOR_URL})
    assert output.origin == "fixture"
    assert output.url == VISITOR_URL
    assert output.title == "Velmora Bridge & Visitor Centre | Harrow Vale"
    assert (output.start_char, output.end_char, output.text_length) == (0, len(text), len(text))
    assert output.text == text
    assert output.content_hash == hashlib.sha256(text.encode()).hexdigest()
    # Offsets index the extracted text, so a later slice lines up with the first.
    later = await _ok(tool, {"url": VISITOR_URL, "start": 40})
    assert later.text == text[40:]
    assert later.content_hash == output.content_hash


async def test_equivalent_urls_reach_the_same_fixture() -> None:
    output = await _ok(
        _fixture_tool(), {"url": "HTTPS://Visit-HarrowVale.example.org:443/velmora-bridge#hours"}
    )
    assert output.url == VISITOR_URL and output.origin == "fixture"


async def test_long_pages_are_paged_in_bounded_slices() -> None:
    lines = [f"Line {i:04d} of the long page." for i in range(200)]
    tool = FetchUrlTool({"https://long.example/": "".join(f"<p>{line}</p>" for line in lines)})
    text = "\n".join(lines)
    first = await _ok(tool, {"url": "https://long.example/"})
    assert (first.start_char, first.end_char) == (0, MAX_FETCH_CHARS)
    pieces = [first.text]
    end = first.end_char
    while end < first.text_length:
        page = await _ok(tool, {"url": "https://long.example/", "start": end})
        assert page.start_char == end
        assert len(page.text) <= MAX_FETCH_CHARS
        pieces.append(page.text)
        end = page.end_char
    assert "".join(pieces) == text
    assert await _error(tool, {"url": "https://long.example/", "start": len(text)}) == (
        "invalid_args",
        "start_out_of_range",
        f"start {len(text)} is past the end of the page text ({len(text)} characters)",
    )


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"url": ""},
        {"url": VISITOR_URL, "start": -1},
        {"url": VISITOR_URL, "start": True},
        {"url": VISITOR_URL, "start": 0.0},
        {"url": VISITOR_URL, "start": "0.0"},
        {"url": VISITOR_URL, "headers": {}},
        {"url": "https://example.org/" + "a" * MAX_URL_CHARS},
    ],
)
async def test_schema_violations_are_invalid_args(args: dict[str, Any]) -> None:
    error_class, reason, _ = await _error(_fixture_tool(), args)
    assert (error_class, reason) == ("invalid_args", "schema_violation")


async def test_integer_string_start_is_accepted() -> None:
    output = await _ok(_fixture_tool(), {"url": TOLL_URL, "start": "5"})
    assert output.start_char == 5


async def test_invalid_url_is_refused() -> None:
    error_class, reason, message = await _error(_fixture_tool(), {"url": "file:///etc/passwd"})
    assert (error_class, reason) == ("invalid_args", "invalid_url")
    assert "only http and https" in message


async def test_unknown_url_with_live_fetch_off() -> None:
    error_class, reason, message = await _error(_fixture_tool(), {"url": LIVE_URL})
    assert (error_class, reason) == ("invalid_args", "unknown_url")
    assert "live fetch is off" in message


async def test_a_page_without_text_is_a_permanent_failure() -> None:
    tool = FetchUrlTool({"https://empty.example/": "<script>x()</script>"})
    error_class, reason, _ = await _error(tool, {"url": "https://empty.example/"})
    assert (error_class, reason) == ("permanent", "empty_page")


def test_output_model_rejects_inconsistent_slices() -> None:
    base: dict[str, Any] = {
        "url": VISITOR_URL,
        "origin": "fixture",
        "title": "",
        "content_hash": "0" * 64,
        "start_char": 0,
        "end_char": 3,
        "text_length": 3,
        "text": "abc",
    }
    FetchUrlOutput.model_validate(base)
    for patch in (
        {"end_char": 4},
        {"start_char": 3},
        {"text": "ab"},
        {"end_char": MAX_FETCH_CHARS + 1, "text_length": MAX_FETCH_CHARS + 1},
    ):
        with pytest.raises(ValueError):
            FetchUrlOutput.model_validate({**base, **patch})


def test_tool_reports_its_pages_and_policy() -> None:
    assert _fixture_tool().fixture_urls == (VISITOR_URL, TOLL_URL)
    assert _fixture_tool().live == LiveFetchPolicy()
    live = LiveFetchPolicy(enabled=True, allowlist=frozenset({LIVE_HOST}))
    assert FetchUrlTool(live=live).live is live
    assert FetchUrlTool().fixture_urls == ()


def test_registries_offer_fetch_url() -> None:
    backend = FixtureSearchBackend(FIXTURE_CORPUS_ID, {"a.md": "Alpha."})
    assert "fetch_url" not in research_registry(backend)
    tool = _fixture_tool()
    assert research_registry(backend, fetch=tool).get("fetch_url") is tool
    assert "fetch_url" in catalog_registry()


async def test_the_catalog_fetch_tool_serves_nothing() -> None:
    executor = ToolExecutor(catalog_registry())
    result = (await executor.execute("fetch_url", {"url": VISITOR_URL}, CTX)).result
    assert result.error is not None and result.error.reason == "unknown_url"


# --- no socket without opt-in -----------------------------------------------------------


@pytest.fixture
def socket_attempts(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Replace the network guard with a recorder: every attempt is logged, then refused."""
    attempts: list[str] = []

    def refuse(kind: str) -> Callable[..., Any]:
        def blocked(*args: Any, **kwargs: Any) -> Any:
            attempts.append(kind)
            raise OSError(f"blocked {kind} in test")

        return blocked

    monkeypatch.setattr(socket.socket, "connect", refuse("connect"))
    monkeypatch.setattr(socket.socket, "connect_ex", refuse("connect_ex"))
    monkeypatch.setattr(socket, "getaddrinfo", refuse("getaddrinfo"))
    monkeypatch.setattr(socket, "create_connection", refuse("create_connection"))
    yield attempts


@pytest.mark.parametrize(
    ("live", "url", "reason"),
    [
        (LiveFetchPolicy(), VISITOR_URL, None),
        (LiveFetchPolicy(), LIVE_URL, "unknown_url"),
        (LiveFetchPolicy(enabled=False, allowlist=frozenset({LIVE_HOST})), LIVE_URL, "unknown_url"),
        (
            LiveFetchPolicy(enabled=True, allowlist=frozenset({"other.example"})),
            LIVE_URL,
            "host_not_allowed",
        ),
        (LiveFetchPolicy(enabled=True), LIVE_URL, "host_not_allowed"),
        (
            LiveFetchPolicy(enabled=True, allowlist=frozenset({LIVE_HOST})),
            "ftp://docs.example.net/",
            "invalid_url",
        ),
    ],
    ids=["fixture", "off", "off-but-listed", "not-listed", "empty-allowlist", "bad-scheme"],
)
async def test_no_socket_is_opened_without_opt_in_and_an_allowlisted_host(
    socket_attempts: list[str], live: LiveFetchPolicy, url: str, reason: str | None
) -> None:
    # Real default transport: nothing stands between the tool and the network but the policy.
    tool = _fixture_tool(live=live)
    result = (await _execute(tool, {"url": url})).result
    assert (None if result.error is None else result.error.reason) == reason
    assert socket_attempts == []


async def test_the_recorder_sees_a_live_fetch_attempt(socket_attempts: list[str]) -> None:
    # Control for the test above: an opted-in, allowlisted host does reach for the network.
    tool = _fixture_tool(live=LiveFetchPolicy(enabled=True, allowlist=frozenset({LIVE_HOST})))
    execution = await _execute(tool, {"url": LIVE_URL})
    error = execution.result.error
    assert error is not None
    assert (error.error_class.value, error.reason) == ("transient", "fetch_failed")
    assert socket_attempts  # DNS lookup (or connect) was attempted, and refused
    assert len(execution.attempts) == 2  # one retry for a transient failure


# --- live fetch (mock transport, no socket) ---------------------------------------------


async def test_live_html_page() -> None:
    server = _Server(_html("<title>Live</title><p>Hello &amp; welcome.</p>"))
    output = await _ok(server.tool(), {"url": LIVE_URL + "#x"})
    assert output.origin == "live"
    assert (output.url, output.title, output.text) == (LIVE_URL, "Live", "Hello & welcome.")
    (request,) = server.requests
    assert str(request.url) == LIVE_URL
    assert request.method == "GET"
    assert request.headers["user-agent"].startswith("RabbitHole/")


async def test_live_plain_text_and_charsets() -> None:
    server = _Server(_html("  plain <b>text</b> \n\n second ", content_type="text/plain"))
    output = await _ok(server.tool(), {"url": LIVE_URL})
    assert output.text == "plain <b>text</b>\nsecond" and output.title == ""

    def latin1(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content="<p>caf\u00e9</p>".encode("latin-1"),
            headers={"content-type": "text/html; charset=ISO-8859-1"},
        )  # fmt: skip

    assert (await _ok(_Server(latin1).tool(), {"url": LIVE_URL})).text == "caf\u00e9"

    def unknown_charset(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content="<p>caf\u00e9</p>".encode(),
            headers={"content-type": "text/html; charset=no-such-codec"},
        )  # fmt: skip

    assert (await _ok(_Server(unknown_charset).tool(), {"url": LIVE_URL})).text == "caf\u00e9"


async def test_a_live_page_is_fetched_once_per_tool() -> None:
    server = _Server(_html("<p>" + "word " * 600 + "</p>"))
    tool = server.tool()
    first = await _ok(tool, {"url": LIVE_URL})
    second = await _ok(tool, {"url": LIVE_URL, "start": first.end_char})
    assert second.content_hash == first.content_hash
    assert len(server.requests) == 1


async def test_fixture_pages_win_over_live_fetch() -> None:
    server = _Server(_html("<p>live copy</p>"))
    tool = server.tool(allow="visit-harrowvale.example.org", pages=load_fixture_pages(PAGES_DIR))
    output = await _ok(tool, {"url": VISITOR_URL})
    assert output.origin == "fixture"
    assert server.requests == []


@pytest.mark.parametrize(
    ("status", "error_class", "attempts"),
    [
        (404, "permanent", 1),
        (403, "permanent", 1),
        (204, "permanent", 1),
        (429, "transient", 2),
        (503, "transient", 2),
    ],
)
async def test_live_http_errors(status: int, error_class: str, attempts: int) -> None:
    server = _Server(_html("<p>x</p>", status=status))
    execution = await _execute(server.tool(), {"url": LIVE_URL})
    error = execution.result.error
    assert error is not None
    assert (error.error_class.value, error.reason) == (error_class, "http_error")
    assert f"HTTP {status}" in error.message
    assert len(server.requests) == attempts


async def test_redirects_are_not_followed() -> None:
    def redirect(_: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://evil.example/" + "x" * 300})

    server = _Server(redirect)
    error_class, reason, message = await _error(server.tool(), {"url": LIVE_URL})
    assert (error_class, reason) == ("permanent", "redirect_not_followed")
    assert "https://evil.example/" in message and len(message) < 600
    assert len(server.requests) == 1  # the redirect target was never requested


@pytest.mark.parametrize("content_type", ["application/json", "image/png", ""])
async def test_unsupported_content_types(content_type: str) -> None:
    def respond(_: httpx.Request) -> httpx.Response:
        headers = {"content-type": content_type} if content_type else {}
        return httpx.Response(200, content=b"{}", headers=headers)

    error_class, reason, message = await _error(_Server(respond).tool(), {"url": LIVE_URL})
    assert (error_class, reason) == ("permanent", "unsupported_content_type")
    assert (content_type or "missing") in message


async def test_oversized_bodies_are_refused() -> None:
    server = _Server(_html("<p>" + "a" * MAX_LIVE_BYTES + "</p>"))
    error_class, reason, _ = await _error(server.tool(), {"url": LIVE_URL})
    assert (error_class, reason) == ("permanent", "page_too_large")


async def test_live_timeouts_and_transport_errors() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    def reset(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("reset", request=request)

    timed_out = _Server(timeout)
    execution = await _execute(timed_out.tool(), {"url": LIVE_URL})
    error = execution.result.error
    assert error is not None and (error.error_class.value, error.reason) == (
        "timeout",
        "fetch_timeout",
    )
    assert len(timed_out.requests) == 1  # timeouts are not retried
    failed = _Server(reset)
    error = (await _execute(failed.tool(), {"url": LIVE_URL})).result.error
    assert error is not None and (error.error_class.value, error.reason) == (
        "transient",
        "fetch_failed",
    )
    assert len(failed.requests) == 2
    # A failed fetch is not cached: the next call tries again.
    failed.respond = _html("<p>back</p>")
    assert (await _ok(failed.tool(), {"url": LIVE_URL})).text == "back"


# --- in a run ---------------------------------------------------------------------------


async def test_a_fetched_page_is_an_observation_but_not_citation_evidence() -> None:
    backend = FixtureSearchBackend.from_directory(FIXTURE_CORPUS, FIXTURE_CORPUS_ID)
    text = html_to_text(load_fixture_pages(PAGES_DIR)[VISITOR_URL]).text
    quote = "It opened in 1986"
    start = text.index(quote)
    page_citation = Citation(
        doc_id=VISITOR_URL[:64], start=start, end=start + len(quote), quote=quote
    )
    loop = AgentLoop(
        OraclePolicy(
            [
                ToolCall(tool="fetch_url", args={"url": VISITOR_URL}),
                FinalAnswer(answer="1986", claims=(Claim(text=quote, citations=(page_citation,)),)),
                GiveUp(reason="cannot_cite"),
            ]
        ),
        ToolExecutor(research_registry(backend, fetch=_fixture_tool())),
        Budgets(max_steps=4, max_repairs=1),
        verifier=CitationVerifier(backend),
        run_id="r-fetch-run",
    )
    state = await loop.run(Task(question="When did the Velmora Bridge visitor centre open?"))
    first = state.steps[0]
    assert first.result is not None and first.result.error is None
    assert quote in state.observations[0].text
    assert EvidenceStore.from_state(state).spans == ()
    verdict = state.steps[1].verdict
    assert verdict is not None and verdict.reason == INVALID_CITATION
    assert state.terminal_label == "abstained:cannot_cite"
