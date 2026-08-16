#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
SUFFIX="$(python3-config --extension-suffix)"

cargo build --manifest-path "$HERE/Cargo.toml" --release --offline
cp "$HERE/target/release/lib_aglm_native.so" "$REPO/aglm_tokenizer/_aglm_native${SUFFIX}"
echo "Built $REPO/aglm_tokenizer/_aglm_native${SUFFIX}"
