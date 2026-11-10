#!/usr/bin/env python3
"""Accuracy of every configuration on an ImageNet validation subset.

Run inside WSL with the project .venv (needs ai_edge_litert, numpy, Pillow):

    python tools/eval/accuracy_sweep.py prep  [N]      # preprocess, FP32 + TFLite-int8 reference
    python tools/eval/accuracy_sweep.py host  [N]      # B0 and B1 kernels as host C, all N images
    python tools/eval/accuracy_sweep.py spike [S]      # B0/B1/X1MAC/X1 on Spike, first S images
    python tools/eval/accuracy_sweep.py report [N] [S] # write results/accuracy.md

Everything is stored in build/acc/ (not committed). Prerequisites: build/gen_target (generated C
for the whole model) and build/model_target (scripts/build/build_model.sh ... B0 B1 X1MAC X1).
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "imagenet_val_subset"
import os
ACC = ROOT / "build" / os.environ.get("ACC_DIR", "acc")
GEN = ROOT / "build" / "gen_target"
TGT = ROOT / "build" / "model_target"
FP32 = ROOT / "models/deit_tiny_patch16_224/deit_tiny_fp32.tflite"
INT8 = ROOT / os.environ.get("ACC_INT8", "models/deit_tiny_patch16_224/deit_tiny_w8a8_pilot.tflite")
OUT_IDX = 776  # output tensor of the generated whole model
SPIKE = Path.home() / "riscv/bin/spike"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def labels(n: int) -> np.ndarray:
    with open(DATA / "labels.csv") as f:
        rows = list(csv.DictReader(f))[:n]
    return np.array([int(r["label"]) for r in rows]), [r["file"] for r in rows]


def interp(path: Path):
    from ai_edge_litert import interpreter as tfl

    it = tfl.Interpreter(
        str(path), experimental_op_resolver_type=tfl.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    )
    it.allocate_tensors()
    return it


def topk(logits: np.ndarray, k: int = 5) -> np.ndarray:
    return np.argsort(-logits.astype(np.int64 if logits.dtype == np.int8 else np.float64),
                      axis=1, kind="stable")[:, :k]


def acc(logits: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    t = topk(logits)
    return float((t[:, 0] == y).mean() * 100), float((t == y[:, None]).any(axis=1).mean() * 100)


# ---------------------------------------------------------------------------------------------
def prep(n: int) -> None:
    y, files = labels(n)
    (ACC / "in").mkdir(parents=True, exist_ok=True)
    fp, q8 = interp(FP32), interp(INT8)
    fin, qin = fp.get_input_details()[0], q8.get_input_details()[0]
    qout = q8.get_output_details()[0]
    assert qout["index"] == OUT_IDX, f"output tensor is {qout['index']}, expected {OUT_IDX}"
    scale, zp = float(qin["quantization_parameters"]["scales"][0]), int(qin["quantization_parameters"]["zero_points"][0])
    flog, qlog = [], []
    for i, fname in enumerate(files):
        img = Image.open(DATA / fname).convert("RGB")
        assert img.size == (224, 224)
        arr = ((np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
        x8 = np.clip(np.round(arr / scale) + zp, -128, 127).astype(np.int8)
        (ACC / "in" / f"{i:05d}.bin").write_bytes(x8.tobytes())
        fp.set_tensor(fin["index"], arr.astype(np.float32)); fp.invoke()
        flog.append(fp.get_tensor(fp.get_output_details()[0]["index"])[0].copy())
        q8.set_tensor(qin["index"], x8); q8.invoke()
        qlog.append(q8.get_tensor(qout["index"])[0].copy())
        if (i + 1) % 100 == 0:
            print(f"prep {i + 1}/{n}", flush=True)
    np.save(ACC / "fp32.npy", np.stack(flog))
    np.save(ACC / "tflite_int8.npy", np.stack(qlog))
    (ACC / "list.txt").write_text("\n".join(str(ACC / "in" / f"{i:05d}.bin") for i in range(n)) + "\n")
    print("FP32 top1/top5:", acc(np.stack(flog), y))
    print("TFLite int8 top1/top5:", acc(np.stack(qlog), y))


def host(n: int) -> None:
    for cfg, extra in (("B0", []), ("B1", [str(ROOT / "src/gen/ops_opt.c")])):
        exe = ACC / f"host_{cfg}"
        if not exe.exists():
          subprocess.run(
            ["gcc", "-O2", f"-I{ROOT / 'src/gen'}", f"-I{GEN}", "-o", str(exe),
             str(ROOT / "tools/eval/host_batch.c"), str(GEN / "model.c"), str(GEN / "weights.c"),
             str(ROOT / "src/gen/ops_ref.c"), *extra],
            check=True,
        )
        # split the image list into chunks and run them in parallel (one process per chunk)
        lines = (ACC / "list.txt").read_text().split()[:n]
        k = 16
        chunks = [lines[i::k] for i in range(k)]  # interleaved: chunk c holds images c, c+k, ...
        procs = []
        for c, ch in enumerate(chunks):
            (ACC / f"list_{c}.txt").write_text("\n".join(ch) + "\n")
            procs.append(subprocess.Popen([str(exe), str(OUT_IDX), str(ACC / f"list_{c}.txt"), str(ACC / f"{cfg}_{c}.bin")]))
        for p in procs:
            assert p.wait() == 0
        got = np.zeros((n, 1000), dtype=np.int8)
        for c in range(k):
            part = np.fromfile(ACC / f"{cfg}_{c}.bin", dtype=np.int8).reshape(-1, 1000)
            got[c::k] = part
        np.save(ACC / f"host_{cfg}.npy", got)
        ref = np.load(ACC / "tflite_int8.npy")[:n]
        print(cfg, "images with logits != TFLite:", int((got != ref).any(axis=1).sum()), "/", n, flush=True)


# ---------------------------------------------------------------------------------------------
def spike_one(args) -> tuple[str, int, dict]:
    cfg, i = args
    work = Path("/tmp/acc") / f"{cfg}_{i}"
    work.mkdir(parents=True, exist_ok=True)
    x = (ACC / "in" / f"{i:05d}.bin").read_bytes()
    g = np.load(ACC / "host_B0.npy")[i].tobytes()  # golden = B0 reference kernels

    def arr(name: str, data: bytes) -> str:
        return f"const int8_t {name}[{len(data)}] = {{" + ",".join(str(b if b < 128 else b - 256) for b in data) + "};\n"

    (work / "model_io.c").write_text(
        '#include "model_io.h"\n' + arr("io_in", x) + arr("io_out", g)
        + f"const model_io_t model_io[] = {{ {{ 0, 0, {len(x)}, 1, io_in }}, {{ {OUT_IDX}, 0, {len(g)}, 0, io_out }} }};\n"
        + "const int model_io_count = 2;\n"
    )
    cc = ["riscv64-unknown-elf-gcc", "-march=rv64imc_zicsr", "-mabi=lp64", "-mcmodel=medany", "-ffreestanding",
          "-nostdlib", "-nostartfiles", "-O2", f"-I{ROOT / 'src/gen'}", f"-I{GEN}"]
    subprocess.run([*cc, "-c", str(work / "model_io.c"), "-o", str(work / "io.o")], check=True)
    c, d = TGT / "common", TGT / cfg
    objs = [c / "start.o", c / "target_main.o", c / "model.o", work / "io.o", c / "weights.o", *sorted(d.glob("ops_*.o"))]
    subprocess.run(["riscv64-unknown-elf-gcc", "-march=rv64imc_zicsr", "-mabi=lp64", "-mcmodel=medany",
                    "-nostdlib", "-nostartfiles", f"-Wl,-T,{ROOT / 'src/startup/gen.ld'}",
                    *map(str, objs), "-o", str(work / "m.elf")], check=True)
    ext = ["--extension=xqmac8"] if cfg.startswith("X1") else []
    env = dict(os.environ, LD_LIBRARY_PATH=f"{Path.home()}/riscv:{Path.home()}/riscv/lib")
    out = subprocess.run([str(SPIKE), "--isa=RV64IMC", *ext, str(work / "m.elf")], capture_output=True, text=True, env=env).stdout
    res = {}
    for line in out.splitlines():
        if line.startswith("CHECK"):
            kv = dict(p.split("=") for p in line.split()[2:] if "=" in p)
            res = {k: int(v) for k, v in kv.items()}
        if line.startswith("TOTAL"):
            res["instret"] = int(line.split("instret=")[1].split()[0])
    subprocess.run(["rm", "-rf", str(work)])
    return cfg, i, res


def spike(s: int) -> None:
    jobs = [(cfg, i) for i in range(s) for cfg in ("X1", "X1MAC", "B1")]
    results: dict = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        for k, (cfg, i, res) in enumerate(ex.map(spike_one, jobs)):
            results.setdefault(cfg, {})[i] = res
            print(f"spike {cfg} img {i}: {res}", flush=True)
            (ACC / "spike.json").write_text(json.dumps(results))


# ---------------------------------------------------------------------------------------------
def report(n: int) -> None:
    y, _ = labels(n)
    fp, q8 = np.load(ACC / "fp32.npy")[:n], np.load(ACC / "tflite_int8.npy")[:n]
    b0, b1 = np.load(ACC / "host_B0.npy")[:n], np.load(ACC / "host_B1.npy")[:n]
    sp = json.loads((ACC / "spike.json").read_text()) if (ACC / "spike.json").exists() else {}

    def row(name: str, g: np.ndarray, ref: np.ndarray | None) -> str:
        t1, t5 = acc(g, y)
        if ref is None:
            diff = agree = "-"
        else:
            diff = f"{int((g != ref).any(axis=1).sum())} (max abs diff {int(np.abs(g.astype(int) - ref.astype(int)).max())})"
            agree = f"{(topk(g)[:, 0] == topk(ref)[:, 0]).mean() * 100:.1f}%"
        return f"| {name} | {len(g)} | {t1:.1f} | {t5:.1f} | {diff} | {agree} |"

    out = ["# Accuracy sweep (ImageNet-1k validation subset)", "",
           f"{n} images from ILSVRC/imagenet-1k validation, 224x224 bicubic, model-card normalization.", "",
           "| Configuration | Images | Top-1 % | Top-5 % | Images with logits != reference | Top-1 agrees with reference |",
           "|---|---|---|---|---|---|",
           row("FP32 DeiT-Tiny", fp, None),
           row("TFLite W8A8 (quantized model)", q8, fp if False else None),
           row("B0 generated C, vs TFLite", b0, q8),
           row("B1 generated C, vs TFLite", b1, q8),
           row("B1 vs B0 (optimization effect)", b1, b0)]
    for cfg in ("B1", "X1MAC", "X1"):
        r = sp.get(cfg)
        if r:
            bad = sum(v["mismatches"] > 0 for v in r.values())
            out.append(f"| {cfg} on Spike, vs B0 | {len(r)} | - | - | {bad} | - |")
    out += ["", "Reference column: TFLite for the B0/B1 rows, B0 for the B1-vs-B0 and Spike rows."]
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "accuracy.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    cmd = sys.argv[1]
    a = [int(v) for v in sys.argv[2:]]
    if cmd == "prep":
        prep(a[0] if a else 1000)
    elif cmd == "host":
        host(a[0] if a else 1000)
    elif cmd == "spike":
        spike(a[0] if a else 20)
    elif cmd == "report":
        report(a[0] if a else 1000)
