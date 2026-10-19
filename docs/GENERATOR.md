# TFLite → Bare-Metal C Generator

Status: **phase 1 complete** (skeleton + first ops, bit-exact on Block 0 MLP + residual).
Model facts live in `docs/MODEL_SUMMARY.md` (regenerate: `python tools/tflite2c/summarize_model.py`).

## Pipeline

```text
.tflite --(tflite_ir.py)--> IR --(gen_model.py: lower + plan)--> build/gen*/{weights.c,model.c,...}
```

| Stage | File | What it does |
|---|---|---|
| Parse | `tools/tflite2c/tflite_ir.py` | Flatbuffer → tensors (shape, quant params, const data incl. external buffers) and ops (kind, options). No TFLite runtime. |
| Lower | `tools/tflite2c/gen_model.py` | Precomputes Q31 multipliers/shifts, GELU LUT, broadcast/transpose strides. |
| Plan | `tools/tflite2c/gen_model.py` | Tensor lifetimes → greedy-by-size arena offsets; RESHAPE aliases its input (no copy). |
| Emit | `tools/tflite2c/gen_model.py` | `weights.c/h` (const data), `model.c` (straight-line `model_run()`), `harness.c` (host test). |
| Kernels | `src/gen/ops_ref.{c,h}`, `src/gen/tflite_math.h` | Scalar reference kernels reproducing TFLite arithmetic. |

## Usage

```bash
# in WSL; needs ~/aca-venv (numpy, ai-edge-litert, tflite, pillow)
bash scripts/run/gen_check.sh 56:60        # op range [start,end): generate, build, compare to TFLite
python tools/tflite2c/gen_model.py --ops 0:648 --out build/gen    # whole model (fails on unsupported ops)
```

`gen_check.sh` dumps every TFLite tensor for `sample.png` (`build/ref/t<idx>.bin`), then builds
the generated C twice on the host and diffs it against TFLite:

- `--no-reuse`: every produced tensor gets its own buffer and is compared (localizes bugs to an op).
- default (reuse): shared arena; only boundary outputs are compared (validates the memory plan).

## Op coverage

| Op | Status |
|---|---|
| FULLY_CONNECTED (per-channel int8) | done, bit-exact |
| GELU (int8 LUT) | done, bit-exact |
| ADD / SUB / MUL (broadcast) | ADD verified; SUB/MUL implemented, not yet exercised |
| RESHAPE | done (alias) |
| TRANSPOSE | implemented, not yet exercised |
| MEAN, SQUARED_DIFFERENCE, RSQRT | **todo** (LayerNorm) |
| BATCH_MATMUL, SOFTMAX, SLICE | **todo** (attention) |
| CONV_2D, CONCATENATION, QUANTIZE | **todo** (patch embed) |

Unsupported ops stop the generator with the op index and kind.

## Findings

- **Rounding.** TFLite's optimized per-channel FULLY_CONNECTED on this host matches
  `srdhm(acc<<left, mult)` followed by a **round-half-up** right shift (`(v + (1<<(r-1))) >> r`),
  not gemmlowp's round-half-away-from-zero. Tried against Block 0 fc1 (197x192x768): gemmlowp
  rounding 404 mismatches, half-up 0. ADD uses the gemmlowp form and matches.
- **Quantize op.** This export stores the input quantize op under opcode slot (0,0); the generator
  maps a one-input ADD to QUANTIZE (matches what the interpreter reports).
- **Constants.** Weights are external flatbuffer buffers (`Buffer.Offset/Size`); the parser reads them.
  Total constants 5.77 MB (INT8 5.69 MB, INT32 88 KB).
- Tensor count is 778 in the flatbuffer; the interpreter reports 822+ (runtime-added tensors). Indices
  below 778 agree.

## Result (phase 1)

Ops 56..59 (fc1 → GELU → fc2 → residual ADD), all 197 tokens, `sample.png`:

| Mode | Tensors compared | Mismatches |
|---|---|---|
| no-reuse | 185, 186, 187, 188 | 0 / 0 / 0 / 0 |
| reuse | 188 | 0 |

Arena: 453,888 B without reuse, 340,416 B with reuse.

## Next phases

2. LayerNorm ops (MEAN, SQUARED_DIFFERENCE, RSQRT, SUB/MUL with broadcast), check ops 8..20.
3. Attention (BATCH_MATMUL, SOFTMAX, SLICE), complete one block.
4. Stem, 12 blocks, head; `rdinstret` per-layer counters; B0/B1/X1 kernel libraries behind the same params.
