# DeiT-Tiny Model Candidate

## Candidate

- Architecture: `deit_tiny_patch16_224`
- Upstream checkpoint: `timm/deit_tiny_patch16_224.fb_in1k`
- Published size/compute: 5.7M parameters, about 1.3 GMAC at 224 x 224
- Upstream checkpoint license: Apache-2.0 (see `NOTICE` and `LICENSE-APACHE-2.0.txt` in this directory)
- Model repository: https://huggingface.co/litert-community/deit_tiny_patch16_224
- Pinned repository revision: `d0677ea57e9dfe6ed8d083d871562435ae90420a`
- Artifact path: `model.tflite` (downloaded locally; ignored by Git)
- Artifact size: 22,978,000 bytes
- Artifact SHA-256: `23d171fd7030e899a29d5d4412dfd3dd14c2cd3d45bec48f2f7fe71c550861fb`

## Inspection Result

Inspected with `tools/model/inspect_tflite.py` and the Python `tflite` FlatBuffer
schema package:

- 855 tensors: 677 FLOAT32 and 178 INT32; no INT8 tensors.
- Input: FLOAT32 `[1, 3, 224, 224]`.
- Output: FLOAT32 `[1, 1000]`.
- Constant storage: 22,869,812 FLOAT32 bytes and 2,560 INT32 bytes.
- Operators include FULLY_CONNECTED, BATCH_MATMUL, GELU, SOFTMAX, MEAN,
  RSQRT, MUL, ADD, SUB, reshape, transpose, and convolution.

Therefore, this LiteRT file is a reference/conversion source, not the required
W8A8 model. A separate Mobilint DeiT repository advertises quantized variants,
but its model card says the compiled artifacts target Mobilint NPU hardware;
those files are not assumed to be portable to RV64IMC.

## Status And Gate

DeiT-Tiny is the primary workload candidate, pending W8A8 feasibility. Before
implementing the full graph, obtain or create a portable W8A8 model and verify:

1. Weight and activation dtypes, scales, and zero-points for each operator.
2. Accuracy against the FLOAT32 source on a fixed validation sample set.
3. Integer/reference parity for two major operators and one complete block.
4. Static weight, activation-buffer, stack, and code-size requirements.
5. Whether normalization, GELU, and softmax can be implemented with acceptable
   integer accuracy and cost on the target.

Until these gates pass, report the model as a candidate rather than claiming
full-model bare-metal support.

The current host conversion state is a validated FP32 LiteRT export and an
integer-only W8A8 pilot. The pilot used only `sample.png` for calibration, so
it is a conversion smoke test and not an accuracy-qualified model.

## Reference And Quantization Pilot

The source reference ran successfully in a temporary Ubuntu CPU environment
using the fixed `sample.png` and the model-card preprocessing. Its top-5 class
indices were 967, 969, 960, 928, and 968. Full logits are saved in the ignored
`reference_output.json` artifact.

The FP32 LiteRT export is saved as `deit_tiny_fp32.tflite` (23,057,032 bytes).
On `sample.png`, its output matched the source model's top-1 class (967) with
maximum absolute logit error `1.15e-05`.

The AI Edge Quantizer `static_wi8_ai8()` pilot is saved as
`deit_tiny_w8a8_pilot.tflite` (6,544,928 bytes, about 3.5x smaller). A LiteRT
runtime audit found 750 INT8 and 72 INT32 tensors, INT8 input/output, and no
operators touching FLOAT32 tensors. With input/output scales applied correctly,
the one-image result had top-1 969 versus source top-1 967, maximum absolute
logit error 1.54, and mean absolute logit error 0.365. This single sample is not
a valid accuracy assessment; obtain a representative calibration set and a
fixed validation set before accepting the model. Export metadata is in the
`export_pilot_report.json` report.

The earlier PyTorch FX smoke test remains in the ignored
`fx_quantization_probe.json` artifact. It inserted dequantization between
floating-point operations and is not the W8A8 pilot described above. The
remaining gates include multi-image calibration/validation, per-operator scale
and zero-point review, integer/reference parity for two major operators and a
complete block, and target implementations for normalization, GELU, and
softmax. Full DeiT inference remains unapproved until those gates pass.

## Host Reference Setup Status

The Ubuntu WSL project venv runs Python 3.12.3 with Torch 2.11.0+cpu,
torchvision 0.26.0+cpu, LiteRT Torch 0.9.4, and AI Edge Quantizer 0.9.0. The
source reference, FP32 export, and W8A8 pilot all ran successfully there. The
standalone `tools/model/inspect_tflite.py` additionally requires the `tflite`
FlatBuffer schema package, which is not installed in this venv; the export
audit above used the LiteRT interpreter instead.