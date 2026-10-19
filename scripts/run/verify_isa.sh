#!/usr/bin/env bash
# Directed + randomized verification of xqmac8 and xrequant.
#   1. host: xrequant oracle vs TFLite fixed-point helpers (20M random cases)
#   2. Spike: every executed custom instruction vs its C oracle (src/workloads/isa_verify_main.c)
# usage: scripts/run/verify_isa.sh [SEED ...]     (default seeds: 1 2 3; ISA_VERIFY_N sets iterations)
# Needs the Spike extension installed (scripts/setup/install_xqmac8_spike.sh).
set -euo pipefail
cd "$(dirname "$0")/../.."
spike_bin="${HOME}/riscv/bin/spike"
export LD_LIBRARY_PATH="${HOME}/riscv:${HOME}/riscv/lib:${LD_LIBRARY_PATH:-}"
out=build/isa_verify
mkdir -p "$out" results

gcc -O2 -Wall -Wextra -Isrc/gen -Isrc/kernels tests/test_xrequant_ref.c -o "$out/test_xrequant_ref"
"$out/test_xrequant_ref"

seeds=("$@")
[ "${#seeds[@]}" -eq 0 ] && seeds=(1 2 3)
riscv64-unknown-elf-gcc -march=rv64imc -mabi=lp64 -mcmodel=medany -nostdlib -nostartfiles -c \
    src/startup/start.S -o "$out/start.o"
rc=0
: > results/isa_verify.log
for seed in "${seeds[@]}"; do
    riscv64-unknown-elf-gcc -march=rv64imc -mabi=lp64 -mcmodel=medany -ffreestanding -nostdlib \
        -nostartfiles -O2 -Wall -Wextra -Werror -DISA_VERIFY_SEED="$seed" ${ISA_VERIFY_N:+-DISA_VERIFY_N=$ISA_VERIFY_N} \
        -Isrc/gen -Isrc/kernels -Wl,-T,src/startup/gen.ld \
        "$out/start.o" src/workloads/isa_verify_main.c src/kernels/xqmac8.c -o "$out/isa_verify_$seed.elf"
    set +e
    "$spike_bin" --isa=RV64IMC --extension=xqmac8 "$out/isa_verify_$seed.elf" | tee -a results/isa_verify.log
    [ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
    set -e
done
exit $rc
