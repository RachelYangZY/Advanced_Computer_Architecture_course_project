#!/usr/bin/env python3
"""Run the local DeiT-Tiny source checkpoint on one fixed image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import timm
import torch
from PIL import Image
from safetensors.torch import load_file
from timm.data import create_transform, resolve_model_data_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    model = timm.create_model(
        "deit_tiny_patch16_224.fb_in1k",
        pretrained=False,
        num_classes=1000,
    )
    state_dict = load_file(str(args.checkpoint), device="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    transform = create_transform(
        **resolve_model_data_config(model),
        is_training=False,
    )
    image = Image.open(args.image).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.inference_mode():
        logits = model(input_tensor)[0]
        probabilities = logits.softmax(dim=0)
        top_probabilities, top_indices = probabilities.topk(5)

    result = {
        "architecture": "deit_tiny_patch16_224.fb_in1k",
        "input_shape": list(input_tensor.shape),
        "input_dtype": str(input_tensor.dtype),
        "top5": [
            {"class_index": int(index), "probability": float(probability), "logit": float(logits[index])}
            for index, probability in zip(top_indices, top_probabilities, strict=True)
        ],
        "logits": [float(value) for value in logits],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "logits"}, indent=2))


if __name__ == "__main__":
    main()
