#!/usr/bin/env bash
set -eu

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
spike_root="${SPIKE_ROOT:-$HOME/riscv-isa-sim}"
customext_dir="$spike_root/customext"
build_dir="$spike_root/build"

if [ ! -f "$customext_dir/customext.mk.in" ]; then
    echo "Spike source not found at $spike_root" >&2
    exit 1
fi

cp "$project_root/src/xqmac8_spike.cc" "$customext_dir/xqmac8_spike.cc"

python3 - "$customext_dir/customext.mk.in" "$build_dir/customext.mk" <<'PY'
import pathlib
import sys

for path_str in sys.argv[1:]:
    p = pathlib.Path(path_str)
    if not p.exists():
        continue
    text = p.read_text()
    if 'xqmac8_spike.cc' in text:
        continue

    old = "        cflush.cc \\\n"
    new = "        cflush.cc \\\n        xqmac8_spike.cc \\\n"
    if old in text:
        text = text.replace(old, new, 1)
    else:
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if 'cflush.cc' in line:
                lines.insert(i + 1, "        xqmac8_spike.cc \\")
                break
        text = '\n'.join(lines) + '\n'
    p.write_text(text)
PY

cd "$build_dir"
make -j"$(nproc)" libcustomext.so
cp libcustomext.so "$HOME/riscv/libcustomext.so"

echo "Built Xqmac8 Spike extension: $HOME/riscv/libcustomext.so"