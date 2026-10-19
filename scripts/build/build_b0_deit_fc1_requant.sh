#!/usr/bin/env bash
set -eu

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="$project_root/build/B0_deit_fc1_requant"
mkdir -p "$build_dir"

for src in int8_gemm.c requantize.c deit_fc1_requant_b0_main.c; do
  riscv64-unknown-elf-gcc \
      -march=rv64imc -mabi=lp64 -mcmodel=medany \
      -ffreestanding -nostdlib -nostartfiles \
      -O0 -Wall -Wextra -Werror -Isrc/kernels -Isrc/workloads \
      -c "$(ls "$project_root"/src/{kernels,workloads}/"$src" 2>/dev/null | head -1)" -o "$build_dir/${src%.c}.o"
done

riscv64-unknown-elf-gcc \
    -march=rv64imc -mabi=lp64 -mcmodel=medany \
    -ffreestanding -nostdlib -nostartfiles \
    -c "$project_root/src/startup/start.S" -o "$build_dir/start.o"

riscv64-unknown-elf-gcc \
    -march=rv64imc -mabi=lp64 -mcmodel=medany \
    -nostdlib -nostartfiles \
    -Wl,-T,"$project_root/src/startup/b0.ld" \
    "$build_dir/start.o" "$build_dir/deit_fc1_requant_b0_main.o" \
    "$build_dir/int8_gemm.o" "$build_dir/requantize.o" \
    -o "$build_dir/b0_deit_fc1_requant.elf"

riscv64-unknown-elf-objdump -d "$build_dir/b0_deit_fc1_requant.elf" \
    >"$build_dir/b0_deit_fc1_requant.dis"
riscv64-unknown-elf-size "$build_dir/b0_deit_fc1_requant.elf"
