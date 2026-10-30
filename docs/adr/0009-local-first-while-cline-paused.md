# ADR-0009: Local-first while Cline is paused

Status: Accepted (2026-10-25). Revisit when the owner re-enables Cline.

## Context
On 2026-10-25 Kim relayed the owner's decision: Cline is off until the owner
says otherwise, and no agent calls the Cline API. RabbitHole's staged plan
used Cline-hosted `cline-pass/deepseek-v4.1-flash` as the default policy
model (controller ADR-016), with local models as a fallback. With Cline
paused, that plan would leave every headline number and the M5 tool-call
check dependent on an unavailable service. Portfolio ADR-008 already
provides the local path. The box is CPU-only, with about 3 GB of RAM to spare
and no swap.

## Decision
- The default agent/policy model is **Qwen3-1.7B, GGUF Q4_K_M, on CPU**
  through an OpenAI-compatible local server (Ollama or llama.cpp
  `llama-server`, per ADR-008). A memory preflight runs before every live
  run.
- The tool-calling support check moves to **M1** and targets this local
  model. A pre-registered decision rule picks native `tool_calls` or the
  JSON-in-text fallback (strict parser, `parse_failures` counter): native is
  the default only with at least 90% schema-valid calls (at least 18/20) and
  a count no lower than the fallback's. The probe file
  (`probes/tool_calls.v1.yaml`, 20 tool-expecting prompts) and the bar
  (`config/probe.toml`, with the probe file's sha256) are committed and
  frozen before the probe runs; the probe refuses to run on a dirty tree or
  a hash mismatch, and any change is a new version, not an in-place edit.
  Results are reported as raw counts (for example `18/20`) plus a 95% Wilson
  interval, with a note that n=20 gives a wide range (18/20 is about
  [0.70, 0.97]), so the probe chooses a format but does not establish
  reliability; `parse_failures` on the full runs does.
- **Every headline number comes from the local model.** Quality thresholds
  are set from a first baseline run on the `dev` split, not promised up
  front. The programmatic graders and eval-hygiene rules are unchanged
  (equal budgets and `k`, untimed warm-up, dirty-tree flag, `smoke-only`
  labels).
- The hosted path is kept only as an **optional OpenAI-compatible backend**
  (configurable base URL, model, and key env var; ADR-016 terms apply if it
  is used). It is not the default. Every hosted run is marked "not run" in
  reports and the README while Cline is paused, and no headline number
  depends on it. If Cline returns, hosted runs are added later as a
  separately labelled extra.
- Cline Pass usage logging, HTTP 402/429 handling, and the ledger live only
  inside the optional backend. The runaway guard is provider-agnostic and
  applies to local runs too.

## Alternatives considered
- **Wait for Cline before starting RabbitHole**: blocks the portfolio on a
  paused service with no date.
- **Keep Cline as the documented default and mark its results "not run"**:
  produces a project whose headline table is empty.
- **Switch to another hosted provider**: not allowed by the owner.

## Consequences
- ADR-0003 is amended to make the local provider the default.
- PROJECT_SPEC sections 2, 4, 5, 7, 8, 9, 10, and 11, and PLAN M1, M5, M6,
  and M7 are updated to match.
- Results will reflect a small CPU model's limits in tool use and multi-hop
  reasoning; this is stated in the README and mitigated by tight tool
  schemas, step budgets, and honest reporting.
- If the owner re-enables Cline, this ADR is revisited. The local results
  remain the headline; hosted results join as a labelled extra unless a new
  ADR changes that.
