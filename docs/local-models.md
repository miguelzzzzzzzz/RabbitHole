# Local models

The default policy model runs on the dev box's CPU behind llama.cpp
`llama-server`, an OpenAI-compatible server (ADR-0009, portfolio ADR-008).
CI never starts it or calls a model. Every pin below is also hard-coded in
[`scripts/local_runtime.sh`](../scripts/local_runtime.sh), and
`tests/test_local_runtime.py` fails if the two disagree.

Installed and checked on the box on 2026-11-02.

## Pins

| Item | Value |
| --- | --- |
| Server | llama.cpp `llama-server`, release `b12032` (`--version`: `0.8.0-dev (build 12032, commit dde879bc4)`) |
| Server asset | `llama-b12032-bin-ubuntu-x64.tar.gz` (CPU build), sha256 `9f10432ca3f66ef003a7a540aaffb9da0acadaefd05404521b9ab7a2436af304` (matches the GitHub release asset digest) |
| Model | Qwen3-1.7B (base model `Qwen/Qwen3-1.7B`, Apache-2.0) |
| GGUF source | `unsloth/Qwen3-1.7B-GGUF` at revision `d7f544eead698dbd1f15126ef60b45a1e1933222`, file `Qwen3-1.7B-Q4_K_M.gguf` |
| GGUF sha256 | `b139949c5bd74937ad8ed8c8cf3d9ffb1e99c866c823204dc42c0d91fa181897` (matches the Hugging Face LFS oid) |
| GGUF size | 1107409472 bytes (1056 MiB) |
| Quantization | Q4_K_M |
| Context length | 4096 (`-c`, `ARP_LOCAL_CTX`); the GGUF metadata gives a training context of 40960 |
| Served model id | `Qwen3-1.7B-Q4_K_M` (`--alias`, `ARP_LOCAL_MODEL`) |
| Endpoint | `http://127.0.0.1:8080/v1` (`ARP_LLM_BASE_URL`), loopback only, no API key |

Why this GGUF: the official `Qwen/Qwen3-1.7B-GGUF` repository publishes only
Q8_0 (1834426016 bytes). Q8_0 weights plus the KV cache plus the 1 GB
preflight headroom do not fit in the roughly 3 GB the box has to spare, so
the Q4_K_M quantization comes from `unsloth/Qwen3-1.7B-GGUF`, pinned by
revision and sha256.

Why llama.cpp and not Ollama: the CPU build is a 22 MB (21.8 MiB) tarball with a
published sha256, and the GGUF file is pinned directly by sha256. Either is
allowed by ADR-0009.

## Install and serve

```bash
scripts/local_runtime.sh install   # download + verify both sha256 sums
scripts/local_runtime.sh verify    # re-check sums and the server version
scripts/local_runtime.sh serve     # verify, then run llama-server in the foreground
```

Files go to `ARP_RUNTIME_DIR` (default `~/.cache/rabbithole/runtime`).
`install` skips files whose sha256 already matches and refuses a download
whose sha256 does not. On the box the runtime lives in
`/workspace/portfolio/runtime`.

`serve` runs:

```
llama-server -m <GGUF> --alias Qwen3-1.7B-Q4_K_M --host 127.0.0.1 --port 8080 \
  -c 4096 -np 1 --jinja --cache-ram 0 --seed 0
```

- `--jinja`: use the model's chat template, needed for native `tools` /
  `tool_calls`. It is the default in this build; passed explicitly anyway.
- `--port 8080`: this build's default port is 9931, so the port is explicit.
- `--cache-ram 0`: this build enables a RAM prompt cache with an 8192 MiB
  limit by default. The box has no swap, so it is turned off.
- `-np 1`: one server slot, so one KV cache.
- Thinking: the chat template has thinking on by default. Non-thinking mode
  is requested per call with
  `"chat_template_kwargs": {"enable_thinking": false}` (the M1 probe and
  live runs do this). A one-message smoke request on 2026-11-02 in that mode
  returned plain content with no `<think>` block.

## Memory

Allocations that `llama-server` reported when loading with the flags above
(its load log, 2026-11-02):

| Buffer | MiB |
| --- | --- |
| Weights repacked for the CPU's AMX kernels (anonymous memory) | 1067.90 |
| Weights served from the memory-mapped GGUF | 243.44 |
| KV cache, f16, 4096 cells x 28 layers (K 224 + V 224) | 448.00 |
| Compute buffer | 60.02 |

So the server needs about 1.8 GiB that the kernel cannot reclaim, which is
more than "GGUF size + KV cache" (1056 + 448 MiB) because the repacked weights
are a second copy. Process RSS also counts the clean page-cache pages of
the mapped GGUF file, which the kernel can drop. `arp doctor` (next M1
step) should budget the reported model, KV, and compute buffers plus the
`ARP_MIN_FREE_MB` headroom, not the GGUF size alone. The 1 GB default
headroom covers the gap today, but the check should not rely on that.

## Box

CPU-only VM shared with other agents: `Intel(R) Xeon(R) Processor`, 8 vCPUs
(AVX-512 and AMX available), 16 GB RAM, no swap, Linux 6.12 x86_64, glibc
2.41. Free memory changes with what else runs on the box, which is why every
live run starts with the memory preflight.
