#!/usr/bin/env bash
# Install and serve the pinned local runtime (docs/local-models.md).
#
#   scripts/local_runtime.sh install   download llama.cpp + the GGUF, verify sha256
#   scripts/local_runtime.sh verify    re-check both sha256 sums, download nothing
#   scripts/local_runtime.sh serve     verify, then exec llama-server (foreground)
#
# Files go to ARP_RUNTIME_DIR (default ~/.cache/rabbithole/runtime). Pins
# live here and in docs/local-models.md; tests/test_local_runtime.py keeps
# the two in sync. Linux x86_64 CPU build only.
set -euo pipefail

LLAMA_BUILD="b12032"
LLAMA_COMMIT="dde879bc4"
LLAMA_ASSET="llama-${LLAMA_BUILD}-bin-ubuntu-x64.tar.gz"
LLAMA_SHA256="9f10432ca3f66ef003a7a540aaffb9da0acadaefd05404521b9ab7a2436af304"
LLAMA_URL="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_BUILD}/${LLAMA_ASSET}"

GGUF_REPO="unsloth/Qwen3-1.7B-GGUF"
GGUF_REVISION="d7f544eead698dbd1f15126ef60b45a1e1933222"
GGUF_FILE="Qwen3-1.7B-Q4_K_M.gguf"
GGUF_SHA256="b139949c5bd74937ad8ed8c8cf3d9ffb1e99c866c823204dc42c0d91fa181897"
GGUF_URL="https://huggingface.co/${GGUF_REPO}/resolve/${GGUF_REVISION}/${GGUF_FILE}"

MODEL_ALIAS="Qwen3-1.7B-Q4_K_M"
HOST="127.0.0.1"
PORT="8080"
CTX="${ARP_LOCAL_CTX:-4096}"

RUNTIME_DIR="${ARP_RUNTIME_DIR:-${HOME}/.cache/rabbithole/runtime}"
LLAMA_DIR="${RUNTIME_DIR}/llama-${LLAMA_BUILD}"
GGUF_PATH="${RUNTIME_DIR}/models/${GGUF_FILE}"

die() { echo "local_runtime: $*" >&2; exit 1; }

sha_ok() { # file expected
  [ -f "$1" ] && [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ]
}

fetch() { # url dest expected_sha
  local url="$1" dest="$2" sha="$3" tmp
  if sha_ok "$dest" "$sha"; then
    echo "ok (cached): $dest"
    return
  fi
  mkdir -p "$(dirname "$dest")"
  tmp="$(mktemp "${dest}.part.XXXXXX")"
  # shellcheck disable=SC2064
  trap "rm -f '$tmp'" EXIT
  curl -fL --retry 3 -o "$tmp" "$url"
  sha_ok "$tmp" "$sha" || die "sha256 mismatch for $url"
  mv "$tmp" "$dest"
  trap - EXIT
  echo "ok (downloaded): $dest"
}

verify() {
  local tarball="${RUNTIME_DIR}/${LLAMA_ASSET}"
  sha_ok "$tarball" "$LLAMA_SHA256" || die "missing or bad ${tarball}; run: $0 install"
  sha_ok "$GGUF_PATH" "$GGUF_SHA256" || die "missing or bad ${GGUF_PATH}; run: $0 install"
  [ -x "${LLAMA_DIR}/llama-server" ] || die "no ${LLAMA_DIR}/llama-server; run: $0 install"
  local version
  version="$("${LLAMA_DIR}/llama-server" --version 2>&1 | grep -m1 '^version:' || true)"
  case "$version" in
    *"build ${LLAMA_BUILD#b}, commit ${LLAMA_COMMIT}"*) ;;
    *) die "unexpected llama-server version: ${version:-<none>}" ;;
  esac
  echo "verified: ${version}; ${GGUF_FILE} sha256 ${GGUF_SHA256}"
}

install() {
  [ "$(uname -s)-$(uname -m)" = "Linux-x86_64" ] || die "only Linux x86_64 is pinned"
  fetch "$LLAMA_URL" "${RUNTIME_DIR}/${LLAMA_ASSET}" "$LLAMA_SHA256"
  if [ ! -x "${LLAMA_DIR}/llama-server" ]; then
    tar -xzf "${RUNTIME_DIR}/${LLAMA_ASSET}" -C "$RUNTIME_DIR"
  fi
  fetch "$GGUF_URL" "$GGUF_PATH" "$GGUF_SHA256"
  verify
}

serve() {
  verify
  # --cache-ram 0: the default prompt cache may grow to 8 GiB of RAM; the box
  # has about 3 GB to spare and no swap. -np 1: one slot, one KV cache.
  # --reasoning-format none: keep <think> in message.content so the probe
  # can score think_block_present (default auto may move thoughts to
  # reasoning_content instead).
  exec "${LLAMA_DIR}/llama-server" \
    -m "$GGUF_PATH" --alias "$MODEL_ALIAS" \
    --host "$HOST" --port "$PORT" \
    -c "$CTX" -np 1 --jinja --cache-ram 0 --seed 0 \
    --reasoning-format none
}

case "${1:-}" in
  install) install ;;
  verify) verify ;;
  serve) serve ;;
  *) echo "usage: $0 {install|verify|serve}" >&2; exit 2 ;;
esac
