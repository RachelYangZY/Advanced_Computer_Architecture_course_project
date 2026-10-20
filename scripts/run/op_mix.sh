#!/usr/bin/env bash
# Per-op instruction mix (stem + transformer block 0) for B0/B1/X1 by differencing Spike PC histograms.
# usage: scripts/run/op_mix.sh        (needs scripts/run/run_model.sh first)
# Output: results/model/op_mix.json, results/model/op_mix.md, docs/figures/{model_map,instr_breakdown}.png
set -euo pipefail
cd "$(dirname "$0")/../.."
source ~/aca-venv/bin/activate
mkdir -p build/opmix
for cfg in B0 B1 X1; do
    python tools/analysis/op_mix.py "build/opmix/$cfg.json" 0 60 "$cfg" > "build/opmix/$cfg.out" 2>&1 &
done
wait
cat build/opmix/*.out
python - <<'EOF'
import json
out = {"first_op": 0, "last_op": 60, "cfgs": {}}
for c in ("B0", "B1", "X1"):
    out["cfgs"].update(json.load(open(f"build/opmix/{c}.json"))["cfgs"])
json.dump(out, open("results/model/op_mix.json", "w"))
EOF
python tools/analysis/plot_model_map.py results/model/op_mix.json docs/figures
