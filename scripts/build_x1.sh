#!/usr/bin/env bash
set -eu

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$project_root/build/X1"
mkdir -p "$build_dir"

common_flags=(
    -march=rv64imc -mabi=lp64 -mcmodel=medany
    -ffreestanding -nostdlib -nostartfiles
    -O2 -Wall -Wextra -Werror -Isrc
)

riscv64-unknown-elf-gcc "${common_flags[@]}" \
    -c "$project_root/src/int8_gemm_x1.c" -o "$build_dir/int8_gemm_x1.o"
riscv64-unknown-elf-gcc "${common_flags[@]}" \
    -c "$project_root/src/x1_main.c" -o "$build_dir/x1_main.o"
riscv64-unknown-elf-gcc "${common_flags[@]}" \
    -c "$project_root/src/start.S" -o "$build_dir/start.o"

riscv64-unknown-elf-gcc \
    -march=rv64imc -mabi=lp64 -mcmodel=medany \
    -nostdlib -nostartfiles \
    -Wl,-T,"$project_root/scripts/b0.ld" \
    "$build_dir/start.o" "$build_dir/x1_main.o" \
    "$build_dir/int8_gemm_x1.o" -o "$build_dir/x1.elf"

riscv64-unknown-elf-objdump -d "$build_dir/x1.elf" > "$build_dir/x1.dis"
riscv64-unknown-elf-size "$build_dir/x1.elf"