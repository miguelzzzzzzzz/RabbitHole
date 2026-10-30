# ADR-0007: Offline-first tools; tool output is untrusted data; citations must be observed

Status: Accepted (2026-10-30)

## Context
Tests and CI must be offline and deterministic. Research agents that read
external content are exposed to prompt injection. Agents can also fabricate
citations.

## Decision
All default tools are offline: corpus search/read, fixture `fetch_url`, and
calculator. Live HTTP fetch requires `ARP_ALLOW_LIVE_FETCH=true` plus a host
allowlist, and is never used in tests (enforced by a socket-blocking
fixture). No tool has side effects. Tool outputs are wrapped in an
untrusted-data envelope in prompts, and a conservative detector flags
instruction-like content. `final_answer` citations must match the
*normalized* stored source text exactly (sliced from `documents.jsonl` at the
cited offsets), carry a matching `content_hash` (else `stale_source`), and
lie within text observed during this run, otherwise the verifier rejects them.

## Alternatives considered
- **Live web search API by default**: costs money or needs keys, and is
  non-reproducible.
- **Trusting model-provided citations**: citations would be unverifiable.
- **An LLM-based injection classifier**: costs a model call per observation
  and is itself unverified.

## Consequences
Results are reproducible and safe to run anywhere. The pattern-based
injection detector will have misses. Its role is measurement
(`followed_injection` rate on injection tasks) and defense in depth, not a
guarantee. The docs state this limitation.
