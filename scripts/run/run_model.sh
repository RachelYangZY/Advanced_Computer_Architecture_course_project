#!/usr/bin/env bash
# Generate, build and run the whole model (or an op range) on Spike for each kernel library,
# then summarize the per-op instruction counts.
# usage: scripts/run/run_model.sh [START:END] [CFG...]      (default 0:648 B0 B1 X1 X1MAC)
#   REF_VARIANT=flip|invert selects another test input (see gen_check.sh).
# Output: results/model/<cfg>.log (raw Spike output), results/model/profile.md
set -euo pipefail
cd "$(dirname "$0")/../.."
source ~/aca-venv/bin/activate

OPS="${1:-0:648}"
shift || true
CFGS=("$@")
[ "${#CFGS[@]}" -eq 0 ] && CFGS=(B0 B1 X1 X1MAC)

VARIANT="${REF_VARIANT:-}"
REF="build/ref${VARIANT:+_$VARIANT}"
GEN="build/gen_target"
OUT="build/model_target"
RES="results/model${VARIANT:+_$VARIANT}"
spike_bin="${HOME}/riscv/bin/spike"
export LD_LIBRARY_PATH="${HOME}/riscv:${HOME}/riscv/lib:${LD_LIBRARY_PATH:-}"

if [ ! -f "$REF/t0.bin" ]; then
    python tools/tflite2c/dump_ref_tensors.py "$REF" $VARIANT 2>&1 | grep -v -i warn
fi
python tools/tflite2c/gen_model.py --ops "$OPS" --out "$GEN" --ref "$REF"
bash scripts/build/build_model.sh "$GEN" "$OUT" "${CFGS[@]}"

mkdir -p "$RES"
logs=()
for cfg in "${CFGS[@]}"; do
    ext=()
    case "$cfg" in X1*) ext=(--extension=xqmac8) ;; esac
    echo "== spike $cfg"
    set +e
    "$spike_bin" --isa=RV64IMC "${ext[@]}" "$OUT/$cfg/model.elf" > "$RES/$cfg.log" 2>&1
    echo "exit=$?" >> "$RES/$cfg.log"
    set -e
    grep -E '^(TOTAL|CHECK|RESULT|exit)' "$RES/$cfg.log"
    logs+=("$cfg=$RES/$cfg.log")
done
python tools/analysis/model_profile.py "${logs[@]}" | tee "$RES/profile.md"
