#!/usr/bin/env python3
"""Prepare a portable W8A8 DeiT-Tiny model for the RV64IMC bare-metal target.

This script is the host-side conversion path that starts from the pinned
DeiT-Tiny checkpoint, exports a FP32 LiteRT reference model, and then attempts
static W8A8 quantization with the documented AI Edge Quantizer recipe.

The workflow intentionally follows the verified host APIs:
  - litert_torch.convert(model.eval(), sample_inputs)
  - edge_model.export(path)
  - recipe.static_wi8_ai8()

It is intentionally written to produce a reproducible export step and an
opportunity to inspect the quantized graph before any bare-metal kernel work.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from safetensors.torch import load_file
from timm.data import create_transform, resolve_model_data_config

# Verified production APIs from the current host environment.
try:
    import litert_torch
except Exception as exc:  # pragma: no cover - environment guard
    litert_torch = None
    LITERT_IMPORT_ERROR = exc
else:
    LITERT_IMPORT_ERROR = None

try:
    from ai_edge_quantizer import recipe
except Exception as exc:  # pragma: no cover - environment guard
    recipe = None
    QUANTIZER_IMPORT_ERROR = exc
else:
    QUANTIZER_IMPORT_ERROR = None


def load_model(checkpoint: Path) -> torch.nn.Module:
    model = timm.create_model("deit_tiny_patch16_224.fb_in1k", pretrained=False, num_classes=1000)
    if checkpoint.exists():
        state = load_file(str(checkpoint), device="cpu")
        model.load_state_dict(state)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model.eval()
    return model


def build_sample_image(image_path: Path, model: torch.nn.Module) -> torch.Tensor:
    transform = create_transform(**resolve_model_data_config(model), is_training=False)
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def export_fp32_litert(model: torch.nn.Module, sample_input: torch.Tensor, output_path: Path) -> None:
    if litert_torch is None:
        raise RuntimeError(f"litert_torch import failed: {LITERT_IMPORT_ERROR}")

    edge_model = litert_torch.convert(model.eval(), (sample_input,))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    edge_model.export(str(output_path))
    print(f"FP32 LiteRT exported to: {output_path}")


def _resolve_tflite_signature(model_path: Path) -> tuple[str, str]:
    try:
        from ai_edge_litert import interpreter as tfl_interpreter
    except Exception as exc:  # pragma: no cover - environment guard
        raise RuntimeError(f"ai_edge_litert interpreter import failed: {exc}") from exc

    interpreter = tfl_interpreter.Interpreter(str(model_path))
    signatures = interpreter.get_signature_list()
    if not signatures:
        raise RuntimeError(f"No TFLite signatures found in {model_path}")

    signature = next(iter(signatures))
    input_names = signatures[signature].get("inputs", [])
    if not input_names:
        raise RuntimeError(f"No inputs found for TFLite signature {signature}")
    return signature, input_names[0]


def build_calibration_set(calibration_dir: Path, model: torch.nn.Module, model_path: Path) -> dict[str, list[dict[str, np.ndarray]]]:
    if not calibration_dir.exists():
        raise FileNotFoundError(f"Calibration directory not found: {calibration_dir}")

    transform = create_transform(**resolve_model_data_config(model), is_training=False)
    signature_key, input_name = _resolve_tflite_signature(model_path)
    calibration_samples: list[dict[str, np.ndarray]] = []
    for image_path in sorted(calibration_dir.iterdir()):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            continue
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).numpy().astype(np.float32)
        calibration_samples.append({input_name: tensor})
    if not calibration_samples:
        raise RuntimeError(f"No calibration images were found in {calibration_dir}")
    return {signature_key: calibration_samples}


def quantize_w8a8(model: torch.nn.Module, fp32_model_path: Path, calibration_dir: Path, output_path: Path) -> None:
    if recipe is None:
        raise RuntimeError(f"ai_edge_quantizer import failed: {QUANTIZER_IMPORT_ERROR}")

    try:
        from ai_edge_quantizer import quantizer as quantizer_api
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"ai_edge_quantizer quantizer API unavailable: {exc}") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    qt = quantizer_api.Quantizer(str(fp32_model_path))
    qt.load_quantization_recipe(recipe.static_wi8_ai8())
    calibration_data = build_calibration_set(calibration_dir, model, fp32_model_path)
    calibration_result = qt.calibrate(calibration_data)
    quantized = qt.quantize(calibration_result, serialize_to_path=str(output_path))
    if quantized.quantized_model is not None:
        print(f"W8A8 quantized model exported to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a W8A8 DeiT-Tiny model for bare-metal RV64IMC work.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the DeiT-Tiny safetensors checkpoint.")
    parser.add_argument("--sample-image", type=Path, required=True, help="Reference ImageNet sample image to build the LiteRT sample input.")
    parser.add_argument("--calibration-dir", type=Path, help="Directory of calibration PNG/JPG images for static W8A8 quantization.")
    parser.add_argument("--fp32-export", type=Path, default=Path("models/deit_tiny_patch16_224/deit_tiny_fp32.tflite"), help="Output path for the FP32 LiteRT export.")
    parser.add_argument("--w8a8-export", type=Path, default=Path("models/deit_tiny_patch16_224/deit_tiny_w8a8.tflite"), help="Output path for the static W8A8 LiteRT export.")
    parser.add_argument("--json", type=Path, help="Optional JSON report path for conversion metadata.")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    sample_input = build_sample_image(args.sample_image, model)
    export_fp32_litert(model, sample_input, args.fp32_export)

    if args.calibration_dir is not None:
        quantize_w8a8(model, args.fp32_export, args.calibration_dir, args.w8a8_export)

    if args.json is not None:
        report = {
            "checkpoint": str(args.checkpoint),
            "sample_image": str(args.sample_image),
            "fp32_export": str(args.fp32_export),
            "w8a8_export": str(args.w8a8_export),
            "calibration_dir": str(args.calibration_dir) if args.calibration_dir else None,
            "recipe": "recipe.static_wi8_ai8()" if recipe is not None else None,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
