#!/usr/bin/env bash
# Build one configuration of a generated DeiT operator tile.
# usage: build_deit_op.sh <op> <B0|B1|X1>
set -eu

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <op> <B0|B1|X1>" >&2
    exit 2
fi

op="$1"
config="$2"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$project_root/build/${config}_op_${op}"
mkdir -p "$build_dir"

# "mlp" is the chained fc1 -> GELU -> fc2 workload; other ops are single tiles.
if [ "$op" = "mlp" ]; then
    scalar_main=deit_mlp_b0_main.c; packed_main=deit_mlp_x1_main.c
else
    scalar_main=deit_op_b0_main.c; packed_main=deit_op_x1_main.c
fi

case "$config" in
    B0) opt="-O0"; srcs=(int8_gemm.c requantize.c "$scalar_main") ;;
    B1) opt="-O2"; srcs=(int8_gemm.c requantize.c "$scalar_main") ;;
    X1) opt="-O2"; srcs=(int8_gemm_x1.c "$packed_main") ;;
    *) echo "unknown config $config" >&2; exit 2 ;;
esac

objs=()
for src in "${srcs[@]}"; do
    riscv64-unknown-elf-gcc \
        -march=rv64imc -mabi=lp64 -mcmodel=medany \
        -ffreestanding -nostdlib -nostartfiles \
        "$opt" -Wall -Wextra -Werror \
        -I"$project_root/src" \
        -DDEIT_OP_HEADER="\"deit_${op}_data.h\"" \
        -c "$project_root/src/$src" -o "$build_dir/${src%.c}.o"
    objs+=("$build_dir/${src%.c}.o")
done

riscv64-unknown-elf-gcc \
    -march=rv64imc -mabi=lp64 -mcmodel=medany \
    -ffreestanding -nostdlib -nostartfiles \
    -c "$project_root/src/start.S" -o "$build_dir/start.o"

elf="$build_dir/${config,,}_${op}.elf"
riscv64-unknown-elf-gcc \
    -march=rv64imc -mabi=lp64 -mcmodel=medany \
    -nostdlib -nostartfiles \
    -Wl,-T,"$project_root/scripts/b0.ld" \
    "$build_dir/start.o" "${objs[@]}" -o "$elf" 2>/dev/null

riscv64-unknown-elf-objdump -d "$elf" > "${elf%.elf}.dis"
echo "== ${config} ${op}: $(riscv64-unknown-elf-size "$elf" | tail -1)"
