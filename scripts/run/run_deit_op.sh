#!/usr/bin/env bash
# Extract, build, run (Spike) and summarize B0/B1/X1 for one DeiT workload.
# usage: run_deit_op.sh <op>     (op: fc1 | qkv | fc2 | mlp)
#   fc1/qkv/fc2: single 8-row x 8-channel linear-layer tile + per-channel requant
#   mlp:         Block 0 MLP chain for one token (fc1 -> requant -> GELU LUT -> fc2)
set -eu

op="${1:?usage: run_deit_op.sh <fc1|qkv|fc2|mlp>}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"

python_bin="${HOME}/aca-venv/bin/python"
spike_bin="${HOME}/riscv/bin/spike"
export LD_LIBRARY_PATH="${HOME}/riscv:${HOME}/riscv/lib:${LD_LIBRARY_PATH:-}"
trace_root="${TRACE_ROOT:-/tmp/aca_traces}"

sed -i 's/\r$//' scripts/*/*.sh

if [ "$op" = "mlp" ]; then
    "$python_bin" tools/extract/extract_deit_mlp_chain.py >/dev/null
    macs="$(grep -E '#define DEIT_MLP_MACS' src/workloads/deit_mlp_data.h | awk '{print $3}')"
    outputs=960   # 768 hidden + 192 final requantized outputs
else
    "$python_bin" tools/extract/extract_deit_op.py --op "$op" >/dev/null
    macs="$(grep -E '#define DEIT_MACS' "src/workloads/deit_${op}_data.h" | awk '{print $3}')"
    outputs=64
fi

for cfg in B0 B1 X1; do
    bash scripts/build/build_deit_op.sh "$op" "$cfg"
done

echo "== Spike correctness (exit 0 = matches golden)"
for cfg in B0 B1 X1; do
    lower="${cfg,,}"
    d="build/${cfg}_op_${op}"
    ext=()
    [ "$cfg" = "X1" ] && ext=(--extension=xqmac8)
    set +e
    "$spike_bin" --isa=RV64IMC "${ext[@]}" "$d/${lower}_${op}.elf"
    echo "${cfg}_${op}_EXIT=$?"
    set -e
    # Traces are large; keep them on the Linux filesystem (fast), not OneDrive.
    mkdir -p "$trace_root/${cfg}_op_${op}"
    "$spike_bin" --isa=RV64IMC "${ext[@]}" -l "$d/${lower}_${op}.elf" \
        >"$trace_root/${cfg}_op_${op}/trace.log" 2>&1 || true
done

mkdir -p results
"$python_bin" tools/analysis/spike_metrics.py --macs "$macs" --outputs "$outputs" \
    "B0=$trace_root/B0_op_${op}/trace.log:build/B0_op_${op}/b0_${op}.elf" \
    "B1=$trace_root/B1_op_${op}/trace.log:build/B1_op_${op}/b1_${op}.elf" \
    "X1=$trace_root/X1_op_${op}/trace.log:build/X1_op_${op}/x1_${op}.elf" \
    | tee "results/deit_${op}_metrics.md"

# The chain's B0 trace is ~1 GB; it is reproducible, so do not keep it.
if [ "$op" = "mlp" ]; then
    rm -rf "$trace_root/B0_op_${op}"
fi
