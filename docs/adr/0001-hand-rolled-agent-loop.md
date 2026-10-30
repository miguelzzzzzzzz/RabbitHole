# ADR-0001: Hand-rolled agent loop instead of an agent framework

Status: Accepted (2026-10-30)

## Context
The project exists to demonstrate agent orchestration: planning, tool
routing, state, budgets, retries/timeouts, guardrails, and tracing. These
are exactly the parts that agent frameworks abstract away. The project also
needs deterministic tests (injected clock, scripted policy), exact replay,
and full control over prompts sent to the model (a small local model,
Qwen3-1.7B, by default; an optional hosted backend as a labelled extra).

## Decision
Implement the loop in-repo as a small state machine (target: a few hundred
lines of orchestration, not counting tools and eval). It is built from the
`Policy`, `Guardrails`, `ToolExecutor`, `CitationVerifier`, and `Tracer`
components, communicates only through typed `Action`/observation models, and
uses `asyncio` for cancellable timeouts. Direct dependencies stay at pydantic,
httpx, numpy, the pinned project-01 package, and optional `rich` for the
trace viewer.

## Alternatives considered
- **LangGraph**: a solid graph/state model with checkpointing. But the graph
  runtime, not the repo, would own retries/state, and debugging and replay
  would go through its abstractions. The heavy LangChain dependency tree
  also cuts against this portfolio's "no framework sprawl" rule.
- **smolagents / PydanticAI / OpenAI Agents SDK**: lighter, but each fixes a
  loop shape and prompt format. Budgets per tool, fault injection, and exact
  record/replay would need workarounds or monkey-patching.
- **CrewAI / AutoGen**: multi-agent focus, out of scope.
- **LlamaIndex agents**: tied to its own retrieval abstractions, which would
  duplicate project 01.

## Consequences
Every behavior a reviewer asks about (when does it stop, what happens on
timeout, what was sent to the model) is answered by reading one module and
its tests. The project re-implements features frameworks provide, which is
intentional and kept small. A short README section maps the loop's
components to LangGraph concepts for readers who know frameworks. Adding a
LangGraph adapter later is possible because the `Policy`/`Tool` interfaces
are framework-neutral.
