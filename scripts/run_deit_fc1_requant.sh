#!/usr/bin/env bash
set -eu

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_bin="${HOME}/aca-venv/bin/python"
spike_bin="${HOME}/riscv/bin/spike"
export LD_LIBRARY_PATH="${HOME}/riscv:${HOME}/riscv/lib:${LD_LIBRARY_PATH:-}"

sed -i 's/\r$//' scripts/*.sh

"$python_bin" scripts/generate_fc1_requant_params.py
gcc -std=c11 -Wall -Wextra -Werror -Isrc \
    src/requantize.c tests/test_requantize.c -o /tmp/test_requantize
/tmp/test_requantize

bash scripts/build_b0_deit_fc1_requant.sh
bash scripts/build_b1_deit_fc1_requant.sh
bash scripts/build_x1_deit_fc1_requant.sh

set +e
"$spike_bin" --isa=RV64IMC build/B0_deit_fc1_requant/b0_deit_fc1_requant.elf
echo "B0_EXIT=$?"
"$spike_bin" --isa=RV64IMC build/B1_deit_fc1_requant/b1_deit_fc1_requant.elf
echo "B1_EXIT=$?"
"$spike_bin" --isa=RV64IMC --extension=xqmac8 \
    build/X1_deit_fc1_requant/x1_deit_fc1_requant.elf
echo "X1_EXIT=$?"
set -e

"$spike_bin" --isa=RV64IMC -l \
    build/B0_deit_fc1_requant/b0_deit_fc1_requant.elf \
    >build/B0_deit_fc1_requant/trace.log 2>&1 || true
"$spike_bin" --isa=RV64IMC -l \
    build/B1_deit_fc1_requant/b1_deit_fc1_requant.elf \
    >build/B1_deit_fc1_requant/trace.log 2>&1 || true
"$spike_bin" --isa=RV64IMC --extension=xqmac8 -l \
    build/X1_deit_fc1_requant/x1_deit_fc1_requant.elf \
    >build/X1_deit_fc1_requant/trace.log 2>&1 || true

"$python_bin" - <<'PY'
from pathlib import Path
import re

LOAD_RE = re.compile(r"\) (?:c\.)?(?:lb|lbu|lh|lhu|lw|ld|lwu)\b")
STORE_RE = re.compile(r"\) (?:c\.)?(?:sb|sh|sw|sd)\b")
MUL_RE = re.compile(r"\) mulw?\b")
XQ_RE = re.compile(r"\) xqmac8\b|0x56c5850b")
CTRL_RE = re.compile(r"\) (?:c\.)?(?:beq|bne|blt|bge|bltu|bgeu|jal|jalr|j|jr)\b")
TOHOST_STORE_RE = re.compile(r"\) (?:c\.)?sd\b")

def bound(lines):
    retired = [ln for ln in lines if ln.startswith("core")]
    for i, ln in enumerate(retired):
        if "c.li" in ln and ", 1" in ln:
            for end in range(i, min(i + 12, len(retired))):
                if TOHOST_STORE_RE.search(retired[end]):
                    return retired[: end + 1]
    cut = len(retired)
    while cut > 0 and (
        "c.j     pc + 0" in retired[cut - 1]
        or "c.j     pc - 2" in retired[cut - 1]
        or "c.nop" in retired[cut - 1]
    ):
        cut -= 1
    return retired[:cut]

def summarize(label, path):
    retired = bound(Path(path).read_text(errors="replace").splitlines())
    return {
        "config": label,
        "retired": len(retired),
        "mulw": sum(bool(MUL_RE.search(ln)) for ln in retired),
        "xqmac8": sum(bool(XQ_RE.search(ln)) for ln in retired),
        "loads": sum(bool(LOAD_RE.search(ln)) for ln in retired),
        "stores": sum(bool(STORE_RE.search(ln)) for ln in retired),
        "control_flow": sum(bool(CTRL_RE.search(ln)) for ln in retired),
    }

rows = [
    summarize("B0", "build/B0_deit_fc1_requant/trace.log"),
    summarize("B1", "build/B1_deit_fc1_requant/trace.log"),
    summarize("X1", "build/X1_deit_fc1_requant/trace.log"),
]
print("| config | retired | mulw | xqmac8 | loads | stores | control_flow |")
print("|---|---:|---:|---:|---:|---:|---:|")
for r in rows:
    print(
        f"| {r['config']} | {r['retired']} | {r['mulw']} | {r['xqmac8']} | "
        f"{r['loads']} | {r['stores']} | {r['control_flow']} |"
    )
# Compare against prior GEMM-only tile if present
prior = Path("build/B1_deit_fc1/trace.log")
if prior.exists():
    gemm_only = summarize("B1_gemm_only", str(prior))
    print(
        f"B1 requant path vs prior B1 GEMM-only retired delta: "
        f"{rows[1]['retired'] - gemm_only['retired']}"
    )
PY
