#!/usr/bin/env bash
set -eu

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

python_bin="${HOME}/aca-venv/bin/python"
spike_bin="${HOME}/riscv/bin/spike"
export LD_LIBRARY_PATH="${HOME}/riscv:${HOME}/riscv/lib:${LD_LIBRARY_PATH:-}"

sed -i 's/\r$//' scripts/*.sh 2>/dev/null || true

"$python_bin" scripts/extract_deit_gemm_tile.py

bash scripts/build_b0_deit_fc1.sh
bash scripts/build_b1_deit_fc1.sh
bash scripts/build_x1_deit_fc1.sh

run_trace() {
  local config="$1"
  local elf="$2"
  local out_dir="$3"
  mkdir -p "$out_dir"
  if [[ "$config" == "X1" ]]; then
    "$spike_bin" --isa=RV64IMC --extension=xqmac8 -l "$elf" \
      >"$out_dir/trace.log" 2>&1 || true
  else
    "$spike_bin" --isa=RV64IMC -l "$elf" >"$out_dir/trace.log" 2>&1 || true
  fi
}

run_trace B0 build/B0_deit_fc1/b0_deit_fc1.elf build/B0_deit_fc1
run_trace B1 build/B1_deit_fc1/b1_deit_fc1.elf build/B1_deit_fc1
run_trace X1 build/X1_deit_fc1/x1_deit_fc1.elf build/X1_deit_fc1

summarize() {
  local label="$1"
  local log="$2"
  local macs="$3"
  python3 - <<PY
from pathlib import Path
log = Path("$log")
lines = log.read_text(errors="replace").splitlines()
retired = [ln for ln in lines if ln.startswith("core")]
# Bound at first tohost completion loop if present; otherwise full retired set.
text = "\n".join(retired)
print(f"[{'$label'}]")
print(f"retired_trace_lines={len(retired)}")
print(f"macs={$macs}")
print(f"mulw={sum(') mulw ' in ln for ln in retired)}")
print(f"loads={sum(any(f') {op} ' in ln for op in ('lb','lbu','lh','lhu','lw','ld')) for ln in retired)}")
print(f"stores={sum(any(f') {op} ' in ln for op in ('sb','sh','sw','sd')) for ln in retired)}")
print(f"xqmac8={sum(('xqmac8' in ln) or ('0x56c5850b' in ln) for ln in retired)}")
print(f"tohost_pass={any('tohost' in ln and '0x0000000000000001' in ln for ln in lines)}")
PY
}


macs=$((8 * 192 * 8))
summarize B0 build/B0_deit_fc1/trace.log "$macs"
summarize B1 build/B1_deit_fc1/trace.log "$macs"
summarize X1 build/X1_deit_fc1/trace.log "$macs"

riscv64-unknown-elf-size build/B0_deit_fc1/b0_deit_fc1.elf
riscv64-unknown-elf-size build/B1_deit_fc1/b1_deit_fc1.elf
riscv64-unknown-elf-size build/X1_deit_fc1/x1_deit_fc1.elf
