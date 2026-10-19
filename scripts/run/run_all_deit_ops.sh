#!/usr/bin/env bash
# Run the generic pipeline for several workloads in sequence.
# usage: run_all_deit_ops.sh [op ...]   (default: fc1 qkv fc2 mlp)
set -eu

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"

ops=("$@")
[ "${#ops[@]}" -eq 0 ] && ops=(fc1 qkv fc2 mlp)

for op in "${ops[@]}"; do
    echo "##### ${op}"
    bash scripts/run/run_deit_op.sh "$op" 2>&1 | grep -v -E '^INFO|^== (B0|B1|X1) '
done
