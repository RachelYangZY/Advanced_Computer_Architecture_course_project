#!/usr/bin/env bash
set -eu

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$project_root/build/B1"
mkdir -p "$build_dir"

riscv64-unknown-elf-gcc \
    -march=rv64imc -mabi=lp64 -mcmodel=medany \
    -ffreestanding -nostdlib -nostartfiles \
    -O2 -Wall -Wextra -Werror \
    -Isrc \
    -c "$project_root/src/int8_gemm.c" -o "$build_dir/int8_gemm.o"

riscv64-unknown-elf-gcc \
    -march=rv64imc -mabi=lp64 -mcmodel=medany \
    -ffreestanding -nostdlib -nostartfiles \
    -O2 -Wall -Wextra -Werror \
    -Isrc \
    -c "$project_root/src/b0_main.c" -o "$build_dir/b0_main.o"

riscv64-unknown-elf-gcc \
    -march=rv64imc -mabi=lp64 -mcmodel=medany \
    -ffreestanding -nostdlib -nostartfiles \
    -c "$project_root/src/start.S" -o "$build_dir/start.o"

riscv64-unknown-elf-gcc \
    -march=rv64imc -mabi=lp64 -mcmodel=medany \
    -nostdlib -nostartfiles \
    -Wl,-T,"$project_root/scripts/b0.ld" \
    "$build_dir/start.o" "$build_dir/b0_main.o" "$build_dir/int8_gemm.o" \
    -o "$build_dir/b1.elf"

riscv64-unknown-elf-objdump -d "$build_dir/b1.elf" > "$build_dir/b1.dis"
riscv64-unknown-elf-size "$build_dir/b1.elf"