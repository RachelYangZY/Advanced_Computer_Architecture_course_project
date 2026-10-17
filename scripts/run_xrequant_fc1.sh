#!/usr/bin/env bash
set -eu

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export LD_LIBRARY_PATH="${HOME}/riscv:${HOME}/riscv/lib:${LD_LIBRARY_PATH:-}"
spike_bin="${HOME}/riscv/bin/spike"
python_bin="${HOME}/aca-venv/bin/python"

sed -i 's/\r$//' scripts/*.sh

gcc -std=c11 -Wall -Wextra -Werror -Isrc \
    tests/test_xrequant_encoding.c -o /tmp/test_xrequant_encoding
/tmp/test_xrequant_encoding

bash scripts/install_xqmac8_spike.sh
cp "${HOME}/riscv/libcustomext.so" "${HOME}/riscv/lib/libcustomext.so" 2>/dev/null || true

"$python_bin" scripts/generate_fc1_requant_params.py
bash scripts/build_b1_deit_fc1_requant.sh
bash scripts/build_x1_deit_fc1_requant.sh

set +e
"$spike_bin" --isa=RV64IMC --extension=xqmac8 \
    build/X1_deit_fc1_requant/x1_deit_fc1_requant.elf
echo "X1_REQUANT_ISA_EXIT=$?"
"$spike_bin" --isa=RV64IMC \
    build/B1_deit_fc1_requant/b1_deit_fc1_requant.elf
echo "B1_SCALAR_REQUANT_EXIT=$?"
set -e

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
XR_RE = re.compile(r"\) xrequant\b|0x56c5950b")
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
        "xrequant": sum(bool(XR_RE.search(ln)) for ln in retired),
        "loads": sum(bool(LOAD_RE.search(ln)) for ln in retired),
        "stores": sum(bool(STORE_RE.search(ln)) for ln in retired),
        "control_flow": sum(bool(CTRL_RE.search(ln)) for ln in retired),
    }

b1 = summarize("B1_scalar_requant", "build/B1_deit_fc1_requant/trace.log")
x1 = summarize("X1_xrequant_isa", "build/X1_deit_fc1_requant/trace.log")
print("| config | retired | mulw | xqmac8 | xrequant | loads | stores | control_flow |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
for r in (b1, x1):
    print(
        f"| {r['config']} | {r['retired']} | {r['mulw']} | {r['xqmac8']} | "
        f"{r['xrequant']} | {r['loads']} | {r['stores']} | {r['control_flow']} |"
    )
if b1["retired"]:
    print(
        f"X1 vs B1 dynamic instruction reduction: "
        f"{100.0 * (1 - x1['retired'] / b1['retired']):.2f}%"
    )
print(f"xrequant count (expect 64): {x1['xrequant']}")
PY
