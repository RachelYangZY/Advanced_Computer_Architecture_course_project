# TFLite → Bare-Metal C Generator

Status: **phase 4 complete** — the whole model (ops 0:648, image → 1000 logits) is generated and the
logits match TFLite exactly on all three test inputs. Arena with reuse: 340,416 B.
Model facts live in `docs/MODEL_SUMMARY.md` (regenerate: `python tools/tflite2c/summarize_model.py`).
Directory roles: `docs/REPO_LAYOUT.md`.

## Pipeline

```text
.tflite --(tflite_ir.py)--> IR --(gen_model.py: lower + plan)--> build/gen*/{weights.c,model.c,...}
```

| Stage | File | What it does |
|---|---|---|
| Parse | `tools/tflite2c/tflite_ir.py` | Flatbuffer → tensors (shape, quant params, const data incl. external buffers) and ops (kind, options). No TFLite runtime. |
| Lower | `tools/tflite2c/gen_model.py` | Precomputes Q31 multipliers/shifts, LUTs (GELU, RSQRT), broadcast/transpose/reduce strides. |
| Plan | `tools/tflite2c/gen_model.py` | Tensor lifetimes → greedy-by-size arena offsets; RESHAPE aliases its input (no copy). |
| Emit | `tools/tflite2c/gen_model.py` | `weights.c/h` (const data), `model.c` (straight-line `model_run()`), `harness.c` (host test). |
| Kernels | `src/gen/ops_ref.{c,h}`, `src/gen/tflite_math.h` | Scalar reference kernels reproducing TFLite arithmetic. |

## Usage

```bash
# in WSL; needs ~/aca-venv (numpy, ai-edge-litert, tflite, pillow)
bash scripts/run/gen_check.sh 44:73        # op range [start,end): generate, build, compare to TFLite
REF_VARIANT=flip bash scripts/run/gen_check.sh 44:73    # same, on a mirrored input (or: invert)
python tools/tflite2c/gen_model.py --ops 0:648 --out build/gen    # whole model (fails on unsupported ops)
```

`gen_check.sh` dumps every TFLite tensor for the test input (`build/ref[_variant]/t<idx>.bin`), then
builds the generated C twice on the host and diffs it against TFLite:

- `--no-reuse`: every produced tensor gets its own buffer and is compared (localizes bugs to an op).
- default (reuse): shared arena; only boundary outputs are compared (validates the memory plan).

Test inputs: `sample.png`, plus a mirrored (`flip`) and colour-inverted (`invert`) copy.

## Op coverage

| Op | Status |
|---|---|
| FULLY_CONNECTED (per-channel int8) | done, bit-exact |
| GELU (int8 LUT) | done, bit-exact |
| ADD / SUB / MUL (broadcast, const scalar operands) | done, bit-exact |
| RESHAPE | done (alias) |
| MEAN (multi-axis, requantizing) | done, bit-exact |
| SQUARED_DIFFERENCE | done, bit-exact |
| RSQRT (int8 LUT) | done, matches with an **empirical +1e-5 bias** (see Findings) |
| TRANSPOSE, SLICE | done (index copies) |
| BATCH_MATMUL (int8 x int8, both activations) | done; rare off-by-one at rounding ties (see Findings) |
| SOFTMAX (int8, last axis) | done, integer-only; rare off-by-one vs TFLite's float LUT (see Findings) |
| CONV_2D (NHWC, per-channel, VALID/SAME) | done, bit-exact (16x16 stride-16 patch embed) |
| CONCATENATION (same quant params) | done, bit-exact |
| QUANTIZE (int8 → int8) | done (copy when params are equal, else requant) |

Unsupported ops stop the generator with the op index and kind.

## Findings

- **Rounding.** TFLite's optimized per-channel FULLY_CONNECTED on this host matches
  `srdhm(acc<<left, mult)` followed by a **round-half-up** right shift (`(v + (1<<(r-1))) >> r`),
  not gemmlowp's round-half-away-from-zero. Tried against Block 0 fc1 (197x192x768): gemmlowp
  rounding 404 mismatches, half-up 0. ADD, MUL, MEAN and SQUARED_DIFFERENCE use the gemmlowp form
  and match (for MUL, half-up gave 2 and 135 mismatches on the two test ops).
- **LayerNorm structure.** TFLite splits it into MEAN → SQUARED_DIFFERENCE → MEAN → ADD(eps) →
  RSQRT → MUL → SUB → MUL → ADD, all INT8 with per-op scales. gamma/beta are folded into the next
  FC, so there is no gamma multiply. MEAN reduces axes [0,2] of a [1,197,192] tensor.
- **SQUARED_DIFFERENCE** uses a left shift of 7 (15 and 20 mismatch: the squared term overflows).
- **RSQRT is not exactly `1/sqrt`.** Building the LUT like TFLite's `PopulateLookupTable` in float32 or
  float64 mismatches TFLite at ~4 of ~1,700 observed (input, output) pairs per input; every mismatch is
  TFLite one step higher, at values within about 0.002 of a rounding boundary. Multiplying the exact value
  by 1.00001 removes all mismatches on all three test inputs (1e-6 and 2e-5 do not). The true algorithm
  is unknown (1e-5 is close to 2^-16, so a 16-bit fixed-point step is a guess). Treat this as a
  calibration constant (`RSQRT_REL_BIAS` in `gen_model.py`), not a derived fact. If a future input
  hits a mismatch, this is the first place to look.
- **Quantize op.** This export stores the input quantize op under opcode slot (0,0); the generator
  maps a one-input ADD to QUANTIZE (matches what the interpreter reports).
- **Constants.** Weights are external flatbuffer buffers (`Buffer.Offset/Size`); the parser reads them.
  Total constants 5.77 MB (INT8 5.69 MB, INT32 88 KB).
- **MEAN rounding (empirical).** TFLite rounds the MEAN result once (exact rational rounding) for 49 of
  the 50 MEAN ops, but matches gemmlowp double rounding for Block 0's LN1 variance mean (input/output
  scale ratio 53.5). The rule used: double rounding if the ratio is < 64, single rounding otherwise.
  The real selection criterion is unknown.
- **SOFTMAX.** TFLite's int8 softmax uses a float32 exp LUT. The target must be integer-only, so the
  kernel uses a Q31 `exp(-scale*d)` table (d = row max − x), a 64-bit row sum and a round-half-up
  integer division. Outputs may differ by one at ties; the harness allows ≤ 0.01% of elements off by
  one for SOFTMAX outputs (prints `ok~(approx)`). None have occurred on the test inputs so far.
- **BATCH_MATMUL.** gemmlowp requantization matches all but one element of ~116k in Block 10's Q·Kᵀ
  (sample and flip inputs): the value lands exactly on a second-stage tie (−103.5); we give −104, TFLite
  −103. Single-stage rounding gives 237 mismatches, so gemmlowp is kept and BMM outputs share the softmax
  tolerance. The difference does not reach the logits.
- Tensor count is 778 in the flatbuffer; the interpreter reports 822+ (runtime-added tensors). Indices
  below 778 agree.

## Results

All ranges compared against TFLite with 0 mismatching tensors, in both no-reuse (every produced tensor)
and reuse (boundary output) modes, on `sample.png`, `flip` and `invert` inputs:

| Ops | Content | Phase |
|---|---|---|
| 56:60 | Block 0 fc1 → GELU → fc2 → residual ADD | 1 |
| 8:20 | Block 0 LayerNorm1 (12 ops) | 2 |
| 44:56 | Block 0 LayerNorm2 | 2 |
| 44:73 | LN2 → MLP → residual → Block 1 LN1 → QKV FC (29 ops) | 2 |
| 20:44 | Block 0 attention (QKV → Q·Kᵀ → softmax → ·V → proj → residual) | 3 |
| 8:60, 60:112 | Block 0, Block 1 | 3 |
| 8:580 | Blocks 0–10 chained | 3 |
| 0:8 | Stem: transpose, patch-embed CONV_2D, cls-token CONCATENATION, pos-embed ADD | 4 |
| **0:648** | **Whole model: image → logits** | 4 |

Ranges with BMM/SOFTMAX include the allowed off-by-one noted under Findings (only t682/t683 ever hit it).
Arena for 0:648: 18,066,328 B without reuse, 340,416 B with reuse.

## Next phases

5. Target build: `main()`, `tohost`, `rdinstret` per-layer counters, B0/B1/X1 kernel libraries behind
   the same param structs, Spike runs, EXPERIMENTS.md.
6. Calibration on more images, host-side FP32-vs-INT8 accuracy, bit-exact parity scripts.

