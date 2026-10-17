#!/usr/bin/env bash
set -eu

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$project_root/build/X1_deit_fc1_requant"
mkdir -p "$build_dir"

common_flags=(
    -march=rv64imc -mabi=lp64 -mcmodel=medany
    -ffreestanding -nostdlib -nostartfiles
    -O2 -Wall -Wextra -Werror -Isrc
)

for src in int8_gemm_x1.c deit_fc1_requant_x1_main.c; do
  riscv64-unknown-elf-gcc "${common_flags[@]}" \
      -c "$project_root/src/$src" -o "$build_dir/${src%.c}.o"
done

riscv64-unknown-elf-gcc "${common_flags[@]}" \
    -c "$project_root/src/start.S" -o "$build_dir/start.o"

riscv64-unknown-elf-gcc \
    -march=rv64imc -mabi=lp64 -mcmodel=medany \
    -nostdlib -nostartfiles \
    -Wl,-T,"$project_root/scripts/b0.ld" \
    "$build_dir/start.o" "$build_dir/deit_fc1_requant_x1_main.o" \
    "$build_dir/int8_gemm_x1.o" \
    -o "$build_dir/x1_deit_fc1_requant.elf"

riscv64-unknown-elf-objdump -d "$build_dir/x1_deit_fc1_requant.elf" \
    >"$build_dir/x1_deit_fc1_requant.dis"
riscv64-unknown-elf-size "$build_dir/x1_deit_fc1_requant.elf"
