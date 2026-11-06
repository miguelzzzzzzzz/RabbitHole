# Providers

Default tool-call format is chosen by the M1 probe under a rule fixed before
the first run (PLAN M1 / ADR-0009). The suite is labelled `smoke-only` and is
**not** a headline number.

## Status

**not run** — `arp probe tool-calls` is implemented; the live local Qwen run
has not been committed yet. `ARP_TOOL_FORMAT` stays empty until that run
records the decision.

## Decision rule

`native` becomes the default only if it is schema-valid on at least 18/20
tool-expecting prompts and its count is not lower than `json`'s; otherwise
the default is `json`.
