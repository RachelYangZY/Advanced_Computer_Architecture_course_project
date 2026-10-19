#!/usr/bin/env python3
"""Summarize checkpoint tensor dtypes and sizes without loading tensors."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from safetensors import safe_open


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    dtype_counts: Counter[str] = Counter()
    dtype_bytes: Counter[str] = Counter()
    tensor_count = 0
    with safe_open(args.checkpoint, framework="np") as checkpoint:
        for name in checkpoint.keys():
            tensor = checkpoint.get_tensor(name)
            dtype = str(tensor.dtype)
            size_bytes = tensor.nbytes
            dtype_counts[dtype] += 1
            dtype_bytes[dtype] += size_bytes
            tensor_count += 1
            print(f"{name}\t{dtype}\t{list(tensor.shape)}\t{size_bytes}")

    print(f"tensor_count={tensor_count}")
    for dtype in sorted(dtype_bytes):
        print(f"dtype={dtype} tensors={dtype_counts[dtype]} bytes={dtype_bytes[dtype]}")


if __name__ == "__main__":
    main()
