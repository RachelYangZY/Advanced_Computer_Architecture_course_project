#!/usr/bin/env bash
# Whole-model dynamic instruction mix for each kernel library, from Spike's PC histogram (spike -g).
# usage: scripts/run/instr_mix.sh [CFG ...]        (default B0 B1 X1 X1MAC; needs run_model.sh first)
# Output: results/model/instr_mix.md. Histograms stay under /tmp (reproducible, not committed).
set -euo pipefail
cd "$(dirname "$0")/../.."
source ~/aca-venv/bin/activate
spike_bin="${HOME}/riscv/bin/spike"
export LD_LIBRARY_PATH="${HOME}/riscv:${HOME}/riscv/lib:${LD_LIBRARY_PATH:-}"

CFGS=("$@")
[ "${#CFGS[@]}" -eq 0 ] && CFGS=(B0 B1 X1 X1MAC)
args=()
for cfg in "${CFGS[@]}"; do
    elf="build/model_target/$cfg/model.elf"
    [ -f "$elf" ] || { echo "missing $elf (run scripts/run/run_model.sh first)" >&2; exit 1; }
    ext=()
    case "$cfg" in X1*) ext=(--extension=xqmac8) ;; esac
    hist="/tmp/aca_hist_$cfg.txt"
    echo "== spike -g $cfg"
    "$spike_bin" -g --isa=RV64IMC "${ext[@]}" "$elf" 2> "$hist" | grep -E '^(TOTAL|RESULT)'
    args+=("$cfg=$elf:$hist")
done
python tools/analysis/instr_mix.py "${args[@]}" | tee results/model/instr_mix.md
