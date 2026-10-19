#!/usr/bin/env bash
# Generate C for an op range, build it on the host, and compare against TFLite tensors.
#   scripts/run/gen_check.sh [START:END]     (default 56:60 = Block 0 MLP + residual)
# Runs twice: --no-reuse (every tensor checked) and with buffer reuse (boundary outputs).
set -euo pipefail
source ~/aca-venv/bin/activate
cd "$(dirname "$0")/../.."

OPS="${1:-56:60}"
REF=build/ref
mkdir -p build

if [ ! -f "$REF/t0.bin" ]; then
    python tools/tflite2c/dump_ref_tensors.py "$REF" 2>&1 | grep -v -i warn
fi

rc=0
for mode in noreuse reuse; do
    out="build/gen_${mode}"
    flag=""
    [ "$mode" = "noreuse" ] && flag="--no-reuse"
    echo "=== $mode ($OPS) ==="
    python tools/tflite2c/gen_model.py --ops "$OPS" --out "$out" $flag
    gcc -O1 -Wall -Wextra -Isrc/gen -I"$out" -o "$out/host_test" \
        "$out/harness.c" "$out/model.c" "$out/weights.c" src/gen/ops_ref.c
    "$out/host_test" "$REF" || rc=1
done
exit $rc
