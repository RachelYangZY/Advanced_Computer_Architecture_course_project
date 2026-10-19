#!/usr/bin/env bash
set -eu

if [ "$#" -ne 4 ]; then
    echo "usage: $0 <trace> <kernel-start-hex> <kernel-end-hex> <mac-count>" >&2
    exit 2
fi

trace="$1"
start="$2"
end="$3"
mac_count="$4"

awk -v start="$start" -v end="$end" -v mac_count="$mac_count" \
    -f "$(dirname "$0")/../../tools/analysis/analyze_kernel_trace.awk" "$trace"
echo "macs=$mac_count"