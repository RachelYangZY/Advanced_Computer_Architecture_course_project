#!/usr/bin/env bash
set -eu

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="$project_root/build/X1_deit_fc1"
mkdir -p "$build_dir"

common_flags=(
    -march=rv64imc -mabi=lp64 -mcmodel=medany
    -ffreestanding -nostdlib -nostartfiles
    -O2 -Wall -Wextra -Werror -Isrc/kernels -Isrc/workloads
)

riscv64-unknown-elf-gcc "${common_flags[@]}" \
    -c "$project_root/src/kernels/int8_gemm_x1.c" -o "$build_dir/int8_gemm_x1.o"
riscv64-unknown-elf-gcc "${common_flags[@]}" \
    -c "$project_root/src/workloads/deit_fc1_x1_main.c" -o "$build_dir/deit_fc1_x1_main.o"
riscv64-unknown-elf-gcc "${common_flags[@]}" \
    -c "$project_root/src/startup/start.S" -o "$build_dir/start.o"

riscv64-unknown-elf-gcc \
    -march=rv64imc -mabi=lp64 -mcmodel=medany \
    -nostdlib -nostartfiles \
    -Wl,-T,"$project_root/src/startup/b0.ld" \
    "$build_dir/start.o" "$build_dir/deit_fc1_x1_main.o" \
    "$build_dir/int8_gemm_x1.o" -o "$build_dir/x1_deit_fc1.elf"

riscv64-unknown-elf-objdump -d "$build_dir/x1_deit_fc1.elf" > "$build_dir/x1_deit_fc1.dis"
riscv64-unknown-elf-size "$build_dir/x1_deit_fc1.elf"
