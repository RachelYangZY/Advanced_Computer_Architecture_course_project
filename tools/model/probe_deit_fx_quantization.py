#!/usr/bin/env python3
"""Probe FX static quantization support for DeiT-Tiny on one calibration image.

This is a conversion/tooling smoke test only. One image is not a valid
calibration set and this does not certify a portable all-integer W8A8 graph.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import timm
import torch
from PIL import Image
from safetensors.torch import load_file
from timm.data import create_transform, resolve_model_data_config
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--backend", default="x86", choices=("x86", "fbgemm"))
    args = parser.parse_args()

    torch.backends.quantized.engine = "x86" if args.backend == "x86" else "fbgemm"
    model = timm.create_model(
        "deit_tiny_patch16_224.fb_in1k", pretrained=False, num_classes=1000
    )
    model.load_state_dict(load_file(str(args.checkpoint), device="cpu"))
    model.eval()

    transform = create_transform(
        **resolve_model_data_config(model), is_training=False
    )
    image = Image.open(args.image).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.inference_mode():
        source_logits = model(input_tensor)
        prepared = prepare_fx(
            model,
            get_default_qconfig_mapping(args.backend),
            (input_tensor,),
        )
        prepared(input_tensor)
        quantized = convert_fx(prepared)
        quantized_logits = quantized(input_tensor)

    quantized_modules = Counter(
        type(module).__name__
        for module in quantized.modules()
        if "Quantized" in type(module).__name__ or "quantized" in type(module).__module__
    )
    quantized_weight_modules = []
    for name, module in quantized.named_modules():
        if hasattr(module, "weight"):
            try:
                if module.weight().is_quantized:
                    quantized_weight_modules.append(name)
            except (TypeError, RuntimeError):
                pass

    report = {
        "backend": args.backend,
        "calibration_images": 1,
        "purpose": "conversion smoke test only; not accuracy-validated W8A8",
        "source_top1": int(source_logits.argmax(dim=1).item()),
        "quantized_top1": int(quantized_logits.argmax(dim=1).item()),
        "max_abs_logit_error": float(
            (source_logits - quantized_logits).abs().max().item()
        ),
        "mean_abs_logit_error": float(
            (source_logits - quantized_logits).abs().mean().item()
        ),
        "quantized_module_types": dict(quantized_modules),
        "quantized_weight_module_count": len(quantized_weight_modules),
        "quantized_weight_modules": quantized_weight_modules,
        "graph": str(quantized.graph),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "graph"}, indent=2))


if __name__ == "__main__":
    main()