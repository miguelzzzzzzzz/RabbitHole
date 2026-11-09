# Providers

Default tool-call format is chosen by the M1 probe under a rule fixed
before the first run (PLAN M1 / ADR-0009). The suite is labelled
`smoke-only` and is **not** a headline number (n=20 → wide Wilson
interval).

## Decision rule

`native` becomes the default only if it is schema-valid on at least 18/20 tool-expecting prompts and its count is not lower than `json`'s; otherwise the default is `json`.

**Chosen default:** `native`

## M1 tool-calling probe (smoke-only)

| Field | Value |
| --- | --- |
| Date | 2026-11-07T09:11:42+08:00 |
| Model | Qwen3-1.7B-Q4_K_M |
| Model GGUF sha256 | `b139949c5bd74937ad8ed8c8cf3d9ffb1e99c866c823204dc42c0d91fa181897` (observed) |
| llama.cpp | 0.8.0-dev (build 12032, commit dde879bc4) (observed) |
| Launch flags | `llama-server -m /workspace/portfolio/runtime/models/Qwen3-1.7B-Q4_K_M.gguf --alias Qwen3-1.7B-Q4_K_M --host 127.0.0.1 --port 8080 -c 4096 -np 1 --jinja --cache-ram 0 --seed 0 --reasoning-format none` (observed) |
| Temperature | 0.0 |
| Seed | 0 |
| Thinking | enable_thinking=False |
| Freeze commit | `fc7512fbf8739ad6e2767fb126d5e8ed89cb83db` |
| Probe sha256 | `e6a6d2c3952a03bb8a2ef0134b269505103a9e2efecade355210ca35e87c1711` |
| Head commit | `b8dd596f82035d9cd06cc8aeaf4ce919cf628861` |
| Rule version | 1 |
| Label | `smoke-only` |

| Format | Schema-valid | 95% Wilson | No-tool correct | Parse failures (25 prompts) |
| --- | --- | --- | --- | --- |
| native | 20/20 | [0.839, 1.000] | 5/5 | 0 (tool 0 / no-tool 0) |
| json | 6/20 | [0.145, 0.519] | 2/5 | 17 (tool 14 / no-tool 3) |

Note: with n=20 the Wilson interval is wide (18/20 ≈ [0.70, 0.97]).
A high schema-valid count shows native is the better format on this
probe set, not that tool calls are "100% reliable". Reliability comes
from parse-failure counts over M2+ runs.

JSON-in-text note: the M1 json system prompt appends the native
"answer directly" instruction after the JSON-only protocol, which
contradicts itself on no-tool prompts. The json score measures this
prompt, not JSON-in-text in the abstract. Fix in `tool_calls.v2` /
`rule_version` 2 before the M6 ablation reuses it (no v2 at M1).

## Memory preflight

```
arp doctor
required RAM: 1819.36 MiB reported buffers + 1024 MiB headroom = 2843.36 MiB
serve flags (observed): -c 4096 --cache-ram 0 -np 1 --host 127.0.0.1 --port 8080 --alias Qwen3-1.7B-Q4_K_M --jinja --seed 0 --reasoning-format none
base URL: http://127.0.0.1:8080/v1
observed pid: 1841259
model sha256 (observed): b139949c5bd74937ad8ed8c8cf3d9ffb1e99c866c823204dc42c0d91fa181897
llama-server --version (observed): 0.8.0-dev (build 12032, commit dde879bc4)
MemAvailable: 4777.05 MiB; SwapTotal: 0.00 MiB
[ok] serve_flags: (observed) context 4096, --cache-ram 0, -np 1, --reasoning-format none, prompt cache disabled
[ok] memory: MemAvailable 4777.05 MiB >= required 2843.36 MiB (buffers 1819.36 + headroom 1024)
[ok] models: http://127.0.0.1:8080/v1/models lists Qwen3-1.7B-Q4_K_M
pass
```


## Both M1 runs (v1 prompts, unchanged bar)

| Run | Date (Asia/Manila) | Launch flags note | native | json | Default |
| --- | --- | --- | --- | --- | --- |
| 1 | 2026-11-06T09:12:44+08:00 | pinned constants in report (pre-observation); server had no `--reasoning-format` (default auto) | 20/20 | 10/20 | native |
| 2 | 2026-11-07T09:11:42+08:00 | observed cmdline includes `--reasoning-format none`; metadata labelled observed | 20/20 | 6/20 | native |

Run 2 decides the default under the frozen rule: native stays default (20 ≥ 18 and 20 ≥ 6).
JSON-in-text remains the fallback; it scored 10/20 on run 1 and 6/20 on run 2 on this
prompt set (see JSON note above). The JSON score moved from 10/20 to 6/20 between runs that used
different `--reasoning-format` settings (default auto, then `none`); two runs cannot support a causal
claim about that setting, so none is made. Reports:
[`m1-toolcall-probe-20261106T011244Z.json`](../evals/results/m1-toolcall-probe-20261106T011244Z.json),
[`m1-toolcall-probe-20261107T011142Z.json`](../evals/results/m1-toolcall-probe-20261107T011142Z.json).
Doctor output for run 2: [`m1-doctor-20261107.txt`](../evals/results/m1-doctor-20261107.txt).
