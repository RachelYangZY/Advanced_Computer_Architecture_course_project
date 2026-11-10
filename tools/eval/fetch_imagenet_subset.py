#!/usr/bin/env python3
"""Stream the first N ImageNet-1k validation images (with labels) into a local folder.

Needs: `hf auth login` done and the terms of ILSVRC/imagenet-1k accepted.

Images are resized to 224x224 (bicubic, same as tools/tflite2c/dump_ref_tensors.py) and saved
as PNG. Labels go to labels.csv as `file,label,label_name`.

Usage:  python tools/eval/fetch_imagenet_subset.py [N=1000] [outdir=data/imagenet_val_subset]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "data/imagenet_val_subset")
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
    names = ds.features["label"].names

    rows = []
    skip = int(sys.argv[3]) if len(sys.argv) > 3 else 0   # first `skip` images are left out
    for i, ex in enumerate(ds):
        if i < skip:
            continue
        if i >= skip + n:
            break
        img = ex["image"].convert("RGB").resize((224, 224), resample=3)  # 3 = BICUBIC
        fname = f"val_{i:05d}.png"
        img.save(out / fname)
        rows.append((fname, ex["label"], names[ex["label"]]))
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{n}", flush=True)

    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "label_name"])
        w.writerows(rows)

    classes = len({r[1] for r in rows})
    print(f"saved {len(rows)} images, {classes} distinct classes -> {out}")


if __name__ == "__main__":
    main()
