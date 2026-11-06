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
| Date | 2026-11-06T09:12:44+08:00 |
| Model | Qwen3-1.7B-Q4_K_M |
| Model GGUF sha256 | `b139949c5bd74937ad8ed8c8cf3d9ffb1e99c866c823204dc42c0d91fa181897` |
| llama.cpp | 0.8.0-dev (build 12032, commit dde879bc4) |
| Launch flags | `llama-server -m <GGUF> --alias Qwen3-1.7B-Q4_K_M --host 127.0.0.1 --port 8080 -c 4096 -np 1 --jinja --cache-ram 0 --seed 0` |
| Temperature | 0.0 |
| Seed | 0 |
| Thinking | enable_thinking=False |
| Freeze commit | `fc7512fbf8739ad6e2767fb126d5e8ed89cb83db` |
| Probe sha256 | `e6a6d2c3952a03bb8a2ef0134b269505103a9e2efecade355210ca35e87c1711` |
| Head commit | `1699f1c0f0af32d3af0f176aca07f4d3366fcd79` |
| Rule version | 1 |
| Label | `smoke-only` |

| Format | Schema-valid | 95% Wilson | No-tool correct | Parse failures |
| --- | --- | --- | --- | --- |
| native | 20/20 | [0.839, 1.000] | 5/5 | 0 |
| json | 10/20 | [0.299, 0.701] | 0/5 | 15 |

Note: with n=20 the Wilson interval is wide (18/20 ≈ [0.70, 0.97]).
Passing the bar means "probably usable", not "90% reliable".
Reliability is measured by `parse_failures` on full M5 runs.

## Memory preflight

```
arp doctor
required RAM: 1819.36 MiB reported buffers + 1024 MiB headroom = 2843.36 MiB
serve flags: -c 4096 --cache-ram 0 -np 1 --host 127.0.0.1 --port 8080 --alias Qwen3-1.7B-Q4_K_M --jinja --seed 0
base URL: http://127.0.0.1:8080/v1
MemAvailable: 6530.43 MiB; SwapTotal: 0.00 MiB
[ok] serve_flags: context 4096, --cache-ram 0, -np 1, prompt cache disabled
[ok] memory: MemAvailable 6530.43 MiB >= required 2843.36 MiB (buffers 1819.36 + headroom 1024)
[ok] models: http://127.0.0.1:8080/v1/models lists Qwen3-1.7B-Q4_K_M
pass
```
