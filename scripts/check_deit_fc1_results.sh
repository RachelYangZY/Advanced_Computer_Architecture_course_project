#!/usr/bin/env bash
set -eu

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export LD_LIBRARY_PATH="${HOME}/riscv:${HOME}/riscv/lib:${LD_LIBRARY_PATH:-}"
spike_bin="${HOME}/riscv/bin/spike"

for d in B0_deit_fc1 B1_deit_fc1 X1_deit_fc1; do
  echo "==== ${d} ===="
  tail -20 "build/${d}/trace.log" || true
  echo
done

echo "=== Spike exit codes (no -l) ==="
set +e
"$spike_bin" --isa=RV64IMC "build/B0_deit_fc1/b0_deit_fc1.elf"
echo "B0_EXIT=$?"
"$spike_bin" --isa=RV64IMC "build/B1_deit_fc1/b1_deit_fc1.elf"
echo "B1_EXIT=$?"
"$spike_bin" --isa=RV64IMC --extension=xqmac8 "build/X1_deit_fc1/x1_deit_fc1.elf"
echo "X1_EXIT=$?"
set -e

python3 - <<'PY'
from pathlib import Path
import re

load_re = re.compile(r"\) (?:c\.)?(?:lb|lbu|lh|lhu|lw|ld|lwu)\b")
store_re = re.compile(r"\) (?:c\.)?(?:sb|sh|sw|sd)\b")
mul_re = re.compile(r"\) mulw?\b")
xq_re = re.compile(r"xqmac8|0x56c5850b")
branch_re = re.compile(r"\) (?:c\.)?(?:beq|bne|blt|bge|bltu|bgeu|jal|jalr|j|jr)\b")

for label, path in [
    ("B0", "build/B0_deit_fc1/trace.log"),
    ("B1", "build/B1_deit_fc1/trace.log"),
    ("X1", "build/X1_deit_fc1/trace.log"),
]:
    lines = Path(path).read_text(errors="replace").splitlines()
    retired = [ln for ln in lines if ln.startswith("core")]
    print(f"[{label}] retired={len(retired)} "
          f"loads={sum(bool(load_re.search(ln)) for ln in retired)} "
          f"stores={sum(bool(store_re.search(ln)) for ln in retired)} "
          f"mul={sum(bool(mul_re.search(ln)) for ln in retired)} "
          f"xqmac8={sum(bool(xq_re.search(ln)) for ln in retired)} "
          f"ctrl={sum(bool(branch_re.search(ln)) for ln in retired)}")
    # Sample a few lines containing load-like ops for X1
    if label == "X1":
        samples = [ln for ln in retired if load_re.search(ln)][:8]
        print("X1 load samples:")
        for ln in samples:
            print(ln)
        # Also show a few middle kernel lines
        print("X1 mid samples:")
        for ln in retired[200:208]:
            print(ln)
PY
