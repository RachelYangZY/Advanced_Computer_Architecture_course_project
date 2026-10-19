#!/usr/bin/env bash
set -eu

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="$project_root/build/Xqmac8Smoke"
mkdir -p "$build_dir"

riscv64-unknown-elf-gcc \
    -march=rv64imc -mabi=lp64 -mcmodel=medany \
    -ffreestanding -nostdlib -nostartfiles \
    -Wall -Wextra -Werror \
    -c "$project_root/tests/xqmac8_smoke.S" -o "$build_dir/xqmac8_smoke.o"

riscv64-unknown-elf-gcc \
    -march=rv64imc -mabi=lp64 -mcmodel=medany \
    -nostdlib -nostartfiles \
    -Wl,-T,"$project_root/src/startup/b0.ld" \
    "$build_dir/xqmac8_smoke.o" \
    -o "$build_dir/xqmac8_smoke.elf"

riscv64-unknown-elf-objdump -d "$build_dir/xqmac8_smoke.elf" > "$build_dir/xqmac8_smoke.dis"
riscv64-unknown-elf-size "$build_dir/xqmac8_smoke.elf"