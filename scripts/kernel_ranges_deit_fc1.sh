#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."

print_range() {
  local elf="$1"
  local symbol="$2"
  echo "ELF=${elf}"
  riscv64-unknown-elf-nm "$elf" | sort | awk -v sym="$symbol" '
    $3 == sym { start=$1; found=1; next }
    found && $2 ~ /[Tt]/ {
      printf "symbol=%s start=0x%s end=0x%s\n", sym, start, $1
      exit
    }
  '
  riscv64-unknown-elf-objdump -d "$elf" | awk -v sym="$symbol" '
    $0 ~ /^[0-9a-f]+ <'"$symbol"'>:/ { in_sym=1; next }
    in_sym && /^[0-9a-f]+ </ { exit }
    in_sym { last=$0 }
    END {
      if (last != "") print "last_line=" last
    }
  '
}

print_range build/B0_deit_fc1/b0_deit_fc1.elf int8_gemm_reference
print_range build/B1_deit_fc1/b1_deit_fc1.elf int8_gemm_reference
print_range build/X1_deit_fc1/x1_deit_fc1.elf int8_gemm_x1
