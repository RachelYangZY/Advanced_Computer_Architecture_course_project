#!/usr/bin/env bash
# Build a generated model (gen_model.py --ref ...) for Spike with one kernel library.
# usage: build_model.sh <gen_dir> <out_dir> <B0|B1|X1|X1MAC>...
#   B0    reference kernels (src/gen/ops_ref.c), TFLite-exact
#   B1    + optimized scalar FC/BMM/CONV (src/gen/ops_opt.c)
#   X1    + xqmac8 dot products and xrequant requantization
#   X1MAC + xqmac8 only, exact software requantization
# OPT (default -O2) sets the optimization level for every configuration.
set -eu

gen_dir="$(realpath "${1:?usage: build_model.sh <gen_dir> <out_dir> <cfg>...}")"
out_dir="${2:?out_dir}"
shift 2
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
opt="${OPT:--O2}"
mkdir -p "$out_dir/common"
out_dir="$(realpath "$out_dir")"

cc() {
    riscv64-unknown-elf-gcc -march=rv64imc_zicsr -mabi=lp64 -mcmodel=medany \
        -ffreestanding -nostdlib -nostartfiles -fno-tree-loop-distribute-patterns \
        "$opt" -Wall -Wextra -Werror -I"$project_root/src/gen" -I"$gen_dir" "$@"
}

# Configuration-independent objects: weights (~6 MB of constants), model, embedded I/O, driver.
for src in "$gen_dir/weights.c" "$gen_dir/model.c" "$gen_dir/model_io.c" "$project_root/src/gen/target_main.c"; do
    obj="$out_dir/common/$(basename "${src%.c}").o"
    if [ ! -f "$obj" ] || [ "$src" -nt "$obj" ]; then
        cc -DMODEL_PROF -c "$src" -o "$obj"
    fi
done
cc -c "$project_root/src/startup/start.S" -o "$out_dir/common/start.o"

for cfg in "$@"; do
    d="$out_dir/$cfg"
    mkdir -p "$d"
    cc -c "$project_root/src/gen/ops_ref.c" -o "$d/ops_ref.o"
    objs=("$d/ops_ref.o")
    case "$cfg" in
        B0) ;;
        B1) cc -c "$project_root/src/gen/ops_opt.c" -o "$d/ops_opt.o"; objs+=("$d/ops_opt.o") ;;
        X1) cc -DGEMM_XQMAC8 -DUSE_XREQUANT -c "$project_root/src/gen/ops_opt.c" -o "$d/ops_opt.o"; objs+=("$d/ops_opt.o") ;;
        X1MAC) cc -DGEMM_XQMAC8 -c "$project_root/src/gen/ops_opt.c" -o "$d/ops_opt.o"; objs+=("$d/ops_opt.o") ;;
        *) echo "unknown config $cfg" >&2; exit 2 ;;
    esac
    riscv64-unknown-elf-gcc -march=rv64imc_zicsr -mabi=lp64 -mcmodel=medany -nostdlib -nostartfiles \
        -Wl,-T,"$project_root/src/startup/gen.ld" \
        "$out_dir/common/start.o" "$out_dir/common/target_main.o" "$out_dir/common/model.o" \
        "$out_dir/common/model_io.o" "$out_dir/common/weights.o" "${objs[@]}" -o "$d/model.elf"
    riscv64-unknown-elf-objdump -d "$d/model.elf" > "$d/model.dis"
    echo "== $cfg: $(riscv64-unknown-elf-size "$d/model.elf" | tail -1)"
done
