#!/usr/bin/env bash
set -euo pipefail

# Build the npm-ready wasm package into ./pkg (ESM, `--target web`).
#
# Usage:
#   ./build.bash                    # name: "web-rwkv-wasm"
#   ./build.bash --scope cryscan    # name: "@cryscan/web-rwkv-wasm"
# Any extra args are forwarded to wasm-pack.

cd "$(dirname "$0")"

echo "==== building wasm (target web) ===="
wasm-pack build --release --target web --out-dir pkg "$@"

echo "==== done: ./pkg ===="
ls -la pkg
