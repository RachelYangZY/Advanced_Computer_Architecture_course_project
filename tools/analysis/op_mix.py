#!/usr/bin/env python3
"""Per-op dynamic instruction mix of the generated model on Spike, by differencing PC histograms.

usage: op_mix.py OUT.json FIRST_OP LAST_OP CFG [CFG ...]
  Needs build/model_target/<CFG>/model.{elf,dis} and results/model/<CFG>.log (run_model.sh).

Method: the PROF counters give each op's instruction count, so the retired-instruction index of every op
boundary is START + cumulative sum (START = instructions before op 0, found by bisection on the first
model_run PC). For each boundary b, `spike -g --instructions=b` prints the PC histogram of the first b
instructions; the difference of two consecutive histograms is exactly the histogram of one op. Each PC is
classified (load/store/alu/multiply/branch/jump/xqmac8/xrequant/...) and attributed to its C function.
"""

from __future__ import annotations

import bisect
import json
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from instr_mix import CLASSES, classify, load_dis, load_hist, load_symbols  # noqa: E402
from model_profile import parse  # noqa: E402

SPIKE = os.path.expanduser("~/riscv/bin/spike")
ENV = dict(os.environ, LD_LIBRARY_PATH=os.path.expanduser("~/riscv") + ":" + os.path.expanduser("~/riscv/lib"))


def run_hist(cfg: str, elf: str, n: int) -> dict[int, int]:
    ext = ["--extension=xqmac8"] if cfg.startswith("X1") else []
    p = subprocess.run([SPIKE, "-g", "--isa=RV64IMC", *ext, f"--instructions={n}", elf],
                       capture_output=True, text=True, env=ENV)
    tmp = f"/tmp/opmix_{cfg}_{n}.txt"
    Path(tmp).write_text(p.stderr)
    h = load_hist(tmp)
    os.remove(tmp)
    return h


def find_start(cfg: str, elf: str, model_run_lo: int, model_run_hi: int) -> int:
    """Smallest n such that the first n instructions include one inside model_run: START = n - 1."""
    def entered(n: int) -> bool:
        return any(model_run_lo <= pc < model_run_hi for pc in run_hist(cfg, elf, n))
    lo, hi = 1, 4_000_000
    while not entered(hi):
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if entered(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo - 1


def main() -> None:
    out_path, first, last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    cfgs = sys.argv[4:]
    result: dict = {"first_op": first, "last_op": last, "cfgs": {}}

    def one(cfg: str) -> tuple[str, dict]:
        elf = f"build/model_target/{cfg}/model.elf"
        insns = load_dis(elf.removesuffix(".elf") + ".dis")
        syms = load_symbols(elf)
        starts = [s[0] for s in syms]
        mr = next(s for s in syms if s[2] == "model_run")
        ops, _, _ = parse(f"results/model/{cfg}.log")
        start = find_start(cfg, elf, mr[0], mr[0] + mr[1])
        bounds = [start]
        for _, _, _, ins, _ in ops:
            bounds.append(bounds[-1] + ins)

        def hist_at(i: int) -> dict[int, int]:
            return run_hist(cfg, elf, bounds[i])

        idx = list(range(first, last + 1))                 # boundary indices first..last (op i spans i -> i+1)
        with ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) // 4)) as ex:
            hs = dict(zip(idx, ex.map(hist_at, idx)))
        per_op = []
        for i in range(first, last):
            a, b = hs[i], hs[i + 1]
            mix, fmix = defaultdict(int), defaultdict(lambda: defaultdict(int))
            for pc, n in b.items():
                d = n - a.get(pc, 0)
                if d == 0 or pc not in insns:
                    continue
                c = classify(*insns[pc])
                k = bisect.bisect_right(starts, pc) - 1
                f = syms[k][2] if k >= 0 else "?"
                mix[c] += d
                fmix[f][c] += d
            _, kind, macs, ins, _ = ops[i]
            tot = sum(mix.values())
            if abs(tot - ins) > 4:           # +2 from the profiling csrr pair is expected
                print(f"WARN {cfg} op {i} {kind}: histogram {tot} vs PROF {ins}", file=sys.stderr)
            per_op.append({"op": i, "kind": kind, "macs": macs, "instret": ins,
                           "mix": {c: mix[c] for c in CLASSES},
                           "funcs": {f: {c: v[c] for c in CLASSES if v[c]} for f, v in fmix.items()}})
        return cfg, {"start": start, "ops": per_op}

    for cfg, data in map(one, cfgs):
        result["cfgs"][cfg] = data
        print(f"{cfg}: START={data['start']}, {len(data['ops'])} ops", flush=True)
    Path(out_path).write_text(json.dumps(result))


if __name__ == "__main__":
    main()
