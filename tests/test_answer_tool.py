"""``final_answer`` as a tool schema (PLAN M3): schema export, parsing, reserved name.

No model calls and no network: the end-to-end runs use `OraclePolicy` over the
fictional fixture corpus.
"""

from __future__ import annotations

import json
from typing import Any

import jsonschema
import pytest

from agentic_research.agent import AgentLoop, Budgets, FinalAnswer, GiveUp, Task, ToolCall
from agentic_research.agent.actions import Citation, Claim
from agentic_research.backends import FixtureSearchBackend
from agentic_research.citations import NO_CITATION, CitationVerifier
from agentic_research.policies.oracle import OraclePolicy
from agentic_research.tools import ToolExecutor, ToolRegistry
from agentic_research.tools.answer import (
    FINAL_ANSWER_TOOL,
    FinalAnswerArgs,
    final_answer_schema,
    parse_final_answer,
)
from agentic_research.tools.base import ErrorClass, ToolContext, ToolError
from agentic_research.tools.builtin import catalog_schemas, default_registry, research_registry
from agentic_research.tools.calculator import CalculatorTool
from agentic_research.tools.registry import (
    RESERVED_TOOL_NAMES,
    TOOL_NAME_PATTERN,
    ToolRegistrationError,
)
from agentic_research.trace.gate import require_citations_verified
from rag_index import FIXTURE_CORPUS, FIXTURE_CORPUS_ID

VELMORA_DOC_ID = "7610eae52aac886e"
DESIGNER = "The bridge was designed by the engineer Ilse Tarkand."
TASK = Task(question="Who designed the Velmora Bridge?")
HASH = "a" * 64


def _citation(**overrides: Any) -> dict[str, Any]:
    citation: dict[str, Any] = {"doc_id": "d1", "start": 0, "end": 5, "quote": "Alpha"}
    citation.update(overrides)
    return citation


def _args(**overrides: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "answer": "Alpha",
        "claims": [{"text": "It is Alpha.", "citations": [_citation()]}],
    }
    args.update(overrides)
    return args


# --- schema ---------------------------------------------------------------------------


def test_schema_is_openai_format_and_valid_json_schema() -> None:
    schema = final_answer_schema()
    assert set(schema) == {"type", "function"}
    assert schema["type"] == "function"
    function = schema["function"]
    assert set(function) == {"name", "description", "parameters"}
    assert function["name"] == FINAL_ANSWER_TOOL == "final_answer"
    assert TOOL_NAME_PATTERN.fullmatch(function["name"])
    assert "cites" in function["description"]
    params = function["parameters"]
    jsonschema.Draft202012Validator.check_schema(params)
    assert params["type"] == "object"
    assert set(params["required"]) == {"answer", "claims"}
    # Unknown keys are refused at every level the model writes.
    assert params["additionalProperties"] is False
    assert params["$defs"]["Claim"]["additionalProperties"] is False
    assert params["$defs"]["Citation"]["additionalProperties"] is False
    assert set(params["$defs"]["Citation"]["required"]) == {"doc_id", "start", "end", "quote"}
    # Every field the model fills in is described.
    for model in ("Claim", "Citation"):
        for name, field in params["$defs"][model]["properties"].items():
            assert field.get("description"), (model, name)
    # Same object as `FinalAnswerArgs`, and a fresh copy each call.
    assert params == FinalAnswerArgs.model_json_schema()
    params["title"] = "changed"
    assert final_answer_schema()["function"]["parameters"]["title"] == "FinalAnswerArgs"


def test_catalog_lists_final_answer_last_and_only_once() -> None:
    names = [s["function"]["name"] for s in catalog_schemas()]
    assert names == ["calculator", "search_corpus", "read_document", "final_answer"]
    assert catalog_schemas()[-1] == final_answer_schema()


VALID: list[tuple[str, dict[str, Any]]] = [
    ("one_cited_claim", _args()),
    ("with_hash", _args(claims=[{"text": "t", "citations": [_citation(content_hash=HASH)]}])),
    ("null_hash", _args(claims=[{"text": "t", "citations": [_citation(content_hash=None)]}])),
    # Well formed; the verifier rejects these as no_citation (with repair feedback).
    ("no_claims", _args(claims=[])),
    ("claim_without_citations", _args(claims=[{"text": "t"}])),
    ("two_claims", _args(claims=[{"text": "a", "citations": [_citation()]}, {"text": "b"}])),
]

INVALID: list[tuple[str, Any]] = [
    ("missing_answer", {"claims": []}),
    ("missing_claims", {"answer": "Alpha"}),
    ("empty_answer", _args(answer="")),
    ("whitespace_answer", _args(answer=" \n\t")),
    ("answer_not_string", _args(answer=3)),
    ("claims_not_list", _args(claims={"text": "t"})),
    ("unknown_top_key", _args(confidence=0.9)),
    ("unknown_claim_key", _args(claims=[{"text": "t", "source": "x"}])),
    ("unknown_citation_key", _args(claims=[{"text": "t", "citations": [_citation(page=1)]}])),
    ("empty_claim_text", _args(claims=[{"text": ""}])),
    (
        "missing_quote",
        _args(claims=[{"text": "t", "citations": [{"doc_id": "d1", "start": 0, "end": 5}]}]),
    ),
    ("negative_start", _args(claims=[{"text": "t", "citations": [_citation(start=-1)]}])),
    ("start_not_int", _args(claims=[{"text": "t", "citations": [_citation(start="zero")]}])),
    ("start_float", _args(claims=[{"text": "t", "citations": [_citation(start=0.5)]}])),
    # Lax mode would read these as integers; offsets refuse them (Chad's 2026-11-27 note).
    ("start_true", _args(claims=[{"text": "t", "citations": [_citation(start=True)]}])),
    ("start_false", _args(claims=[{"text": "t", "citations": [_citation(start=False)]}])),
    ("end_true", _args(claims=[{"text": "t", "citations": [_citation(start=0, end=True)]}])),
    ("start_float_string", _args(claims=[{"text": "t", "citations": [_citation(start="0.0")]}])),
    ("end_underscore_string", _args(claims=[{"text": "t", "citations": [_citation(end="1_0")]}])),
    ("start_padded_string", _args(claims=[{"text": "t", "citations": [_citation(start=" 0")]}])),
    ("empty_doc_id", _args(claims=[{"text": "t", "citations": [_citation(doc_id="")]}])),
    ("long_doc_id", _args(claims=[{"text": "t", "citations": [_citation(doc_id="d" * 65)]}])),
    ("bad_hash", _args(claims=[{"text": "t", "citations": [_citation(content_hash="ABC")]}])),
]


@pytest.mark.parametrize(("case", "arguments"), VALID, ids=[c for c, _ in VALID])
def test_schema_and_parser_accept_the_same_valid_calls(
    case: str, arguments: dict[str, Any]
) -> None:
    validator = jsonschema.Draft202012Validator(final_answer_schema()["function"]["parameters"])
    assert validator.is_valid(arguments), case
    answer = parse_final_answer(arguments)
    assert isinstance(answer, FinalAnswer), case
    assert answer.answer == arguments["answer"]
    assert len(answer.claims) == len(arguments["claims"])


@pytest.mark.parametrize(("case", "arguments"), INVALID, ids=[c for c, _ in INVALID])
def test_schema_and_parser_refuse_the_same_invalid_calls(case: str, arguments: Any) -> None:
    validator = jsonschema.Draft202012Validator(final_answer_schema()["function"]["parameters"])
    assert not validator.is_valid(arguments), case
    error = parse_final_answer(arguments)
    assert isinstance(error, ToolError), case
    assert (error.error_class, error.reason) == (ErrorClass.INVALID_ARGS, "schema_violation")
    assert error.message.startswith("invalid arguments for 'final_answer': ")


def test_end_not_after_start_is_refused_by_the_parser_only() -> None:
    # JSON Schema cannot compare two fields; the Citation model does (documented).
    arguments = _args(claims=[{"text": "t", "citations": [_citation(start=5, end=5)]}])
    validator = jsonschema.Draft202012Validator(final_answer_schema()["function"]["parameters"])
    assert validator.is_valid(arguments)
    error = parse_final_answer(arguments)
    assert isinstance(error, ToolError)
    assert error.reason == "schema_violation"
    assert "end must be > start" in error.message


def test_the_parser_accepts_integer_strings_like_every_tool() -> None:
    # Arguments are validated in pydantic's lax mode, as for every registered tool:
    # an offset written as "0" is read as 0 although the schema says integer. The
    # verifier still checks the quote at the parsed offsets.
    arguments = _args(claims=[{"text": "t", "citations": [_citation(start="0", end="5")]}])
    validator = jsonschema.Draft202012Validator(final_answer_schema()["function"]["parameters"])
    assert not validator.is_valid(arguments)
    answer = parse_final_answer(arguments)
    assert isinstance(answer, FinalAnswer)
    citation = answer.claims[0].citations[0]
    assert (citation.start, citation.end) == (0, 5)


@pytest.mark.parametrize(
    ("start", "end"), [(0.0, 5), (0, 5.0), (0.0, 5.0)], ids=["start", "end", "both"]
)
def test_integral_floats_are_refused_by_the_parser_only(start: float, end: float) -> None:
    # JSON Schema's "integer" accepts a number with a zero fractional part (0.0);
    # the parser refuses every float, so an offset is never a coerced float.
    arguments = _args(claims=[{"text": "t", "citations": [_citation(start=start, end=end)]}])
    validator = jsonschema.Draft202012Validator(final_answer_schema()["function"]["parameters"])
    assert validator.is_valid(arguments)
    error = parse_final_answer(arguments)
    assert isinstance(error, ToolError)
    assert error.reason == "schema_violation"
    assert "not a float" in error.message


def test_json_floats_and_booleans_are_refused_from_a_native_call() -> None:
    # Native tool calls carry JSON text: 0.0 and true decode to float and bool.
    for start, expected in (("0.0", "not a float"), ("true", "not a boolean")):
        raw = (
            '{"answer": "Alpha", "claims": [{"text": "t", "citations": '
            f'[{{"doc_id": "d1", "start": {start}, "end": 5, "quote": "Alpha"}}]}}]}}'
        )
        error = parse_final_answer(raw)
        assert isinstance(error, ToolError), start
        assert error.reason == "schema_violation"
        assert expected in error.message


def test_citation_offsets_refuse_coercible_values_directly() -> None:
    # The rule lives on `Citation`, so a FinalAnswer built any other way gets it too.
    for bad in (True, False, 0.0, 2.5, "0.0", "+0", "1_0", " 0", ""):
        with pytest.raises(ValueError, match="start"):
            Citation(doc_id="d1", start=bad, end=5, quote="Alpha")  # type: ignore[arg-type]
    assert Citation(doc_id="d1", start="0", end="5", quote="Alpha").start == 0  # type: ignore[arg-type]


# --- parsing ----------------------------------------------------------------------------


def test_parses_a_native_json_string_into_a_final_answer() -> None:
    raw = json.dumps(_args(claims=[{"text": "t", "citations": [_citation(content_hash=HASH)]}]))
    answer = parse_final_answer(raw)
    assert answer == FinalAnswer(
        answer="Alpha",
        claims=(
            Claim(
                text="t",
                citations=(
                    Citation(doc_id="d1", start=0, end=5, quote="Alpha", content_hash=HASH),
                ),
            ),
        ),
    )


def test_answer_text_is_kept_verbatim() -> None:
    answer = parse_final_answer(_args(answer="  Ilse Tarkand \n"))
    assert isinstance(answer, FinalAnswer)
    assert answer.answer == "  Ilse Tarkand \n"


def test_a_final_answer_round_trips_through_the_tool_arguments() -> None:
    original = FinalAnswer(
        answer="Alpha",
        claims=(
            Claim(text="a", citations=(Citation(doc_id="d1", start=0, end=5, quote="Alpha"),)),
            Claim(text="b"),
        ),
    )
    arguments = original.model_dump(mode="json", exclude={"kind"})
    jsonschema.validate(arguments, final_answer_schema()["function"]["parameters"])
    assert parse_final_answer(json.dumps(arguments)) == original


@pytest.mark.parametrize(
    ("arguments", "reason"),
    [
        ("{not json", "invalid_json"),
        ('{"answer": "a", "answer": "b", "claims": []}', "duplicate_key"),
        ("[" * 100_000, "invalid_json"),
        ('["Alpha"]', "not_an_object"),
        ("null", "not_an_object"),
        (None, "schema_violation"),
        ("", "schema_violation"),
        ("   ", "schema_violation"),
    ],
)
def test_malformed_arguments_get_the_registry_reason_codes(
    arguments: str | None, reason: str
) -> None:
    error = parse_final_answer(arguments)
    assert isinstance(error, ToolError)
    assert (error.error_class, error.reason) == (ErrorClass.INVALID_ARGS, reason)
    assert "'final_answer'" in error.message


# --- reserved name ------------------------------------------------------------------------


class _AnswerTool(CalculatorTool):
    name = "final_answer"


def test_the_registry_refuses_a_tool_named_final_answer() -> None:
    assert FINAL_ANSWER_TOOL in RESERVED_TOOL_NAMES
    with pytest.raises(ToolRegistrationError, match="reserved for the answer action"):
        ToolRegistry([_AnswerTool()])
    registry = default_registry()
    with pytest.raises(ToolRegistrationError, match="reserved"):
        registry.register(_AnswerTool())
    assert registry.names == ["calculator"]


@pytest.mark.parametrize("arguments", [None, {}, _args(), json.dumps(_args())])
def test_the_registry_never_runs_final_answer(arguments: Any) -> None:
    for registry in (ToolRegistry(), default_registry()):
        error = registry.validate_args("final_answer", arguments)
        assert isinstance(error, ToolError)
        assert (error.error_class, error.reason) == (ErrorClass.INVALID_ARGS, "reserved_tool_name")
        assert "ends the run" in error.message
        assert "final_answer" not in registry


async def test_the_executor_refuses_final_answer_without_running_anything() -> None:
    execution = await ToolExecutor(default_registry()).execute(
        "final_answer", _args(), ToolContext(run_id="r", step=0)
    )
    error = execution.result.error
    assert error is not None and error.reason == "reserved_tool_name"
    assert execution.attempts == ()  # refused before any attempt


# --- end to end over the fixture corpus ------------------------------------------------


def _fixture() -> FixtureSearchBackend:
    return FixtureSearchBackend.from_directory(FIXTURE_CORPUS, FIXTURE_CORPUS_ID)


def _designer_arguments(backend: FixtureSearchBackend) -> str:
    """A model's native ``final_answer`` call: a JSON string citing the designer sentence."""
    doc = backend.document(VELMORA_DOC_ID)
    at = doc.text.index(DESIGNER)
    citation = {
        "doc_id": VELMORA_DOC_ID,
        "start": at,
        "end": at + len(DESIGNER),
        "quote": DESIGNER,
        "content_hash": doc.content_hash,
    }
    return json.dumps(
        {"answer": "Ilse Tarkand", "claims": [{"text": DESIGNER, "citations": [citation]}]}
    )


async def test_a_parsed_final_answer_call_is_accepted_by_the_citation_verifier() -> None:
    backend = _fixture()
    answer = parse_final_answer(_designer_arguments(backend))
    assert isinstance(answer, FinalAnswer)
    loop = AgentLoop(
        OraclePolicy(
            [
                ToolCall(tool="search_corpus", args={"query": "Velmora Bridge designed", "k": 3}),
                answer,
            ]
        ),
        ToolExecutor(research_registry(backend)),
        Budgets(max_steps=3, max_repairs=0),
        verifier=CitationVerifier(backend),
        run_id="r-final-answer-tool",
    )
    traced = await loop.run_traced(TASK)
    assert traced.state.terminal_label == "succeeded"
    assert traced.state.answer == "Ilse Tarkand"
    assert require_citations_verified(traced.trace).verifier == "citations"


async def test_a_policy_that_calls_final_answer_as_a_tool_gets_an_observation() -> None:
    # A policy must turn final_answer into a FinalAnswer; sent as a ToolCall it ends
    # nothing and grounds nothing: the step is an invalid_args result the run moves past.
    backend = _fixture()
    misrouted = ToolCall(tool="final_answer", args=json.loads(_designer_arguments(backend)))
    loop = AgentLoop(
        OraclePolicy([misrouted, FinalAnswer(answer="Ilse Tarkand"), GiveUp(reason="done")]),
        ToolExecutor(research_registry(backend)),
        Budgets(max_steps=4, max_repairs=1),
        verifier=CitationVerifier(backend),
        run_id="r-misrouted",
    )
    state = await loop.run(TASK)
    first = state.steps[0]
    assert first.result is not None and first.result.error is not None
    assert first.result.error.reason == "reserved_tool_name"
    assert "ends the run" in state.observations[0].text
    # The uncited answer that follows is still checked by the verifier.
    second = state.steps[1].verdict
    assert second is not None and second.reason == NO_CITATION
    assert state.terminal_label == "abstained:done"
