## Configuration Comparison

All performance claims use the same benchmark input, matrix shapes, model data,
compiler settings, and measurement method. Configurations are kept separate so
an earlier result can be rerun after later changes.

| ID | Processor/software configuration | Required status |
|---|---|---|
| B0 | RV64IMC with straightforward scalar kernels | Baseline |
| B1 | RV64IMC with software-optimized kernels | Required comparison |
| X1 | RV64IMC plus Xqmac and requantization support | Required extension |
| X2 | Optional extension selected after profiling X1 | Optional |

Each configuration must have its own build output directory:

```text
build/B0/
build/B1/
build/X1/
build/X2/
```

The benchmark runner must accept a configuration ID and never overwrite another
configuration's binary or measurements. Results should include the source
revision, compiler flags, simulator settings, input-data identifier, and date.
If a benchmark is missing, rerun the same configuration with the same manifest
instead of reconstructing it from memory.

## Model Selection And Quantization

The model is the application workload; the GEMM and other operators are the
computational kernels executed by the processor. A transformer is acceptable
because its inference can be quantized to W8A8: INT8 weights and INT8
activations, with wider integer accumulators. The target program must not depend
on floating-point hardware.
The model-selection check is:

1. Obtain a quantized LiteRT/TFLite model or quantize a model offline.
2. Verify its operator set and tensor types are supported by the planned
   integer kernels.
3. Compare a small extracted operator against a host-side integer reference.
4. Select a CNN fallback such as MobileNetV3-Large if transformer conversion
   or integer operator support becomes a project risk.

### Primary Workload Candidate: DeiT-Tiny

DeiT-Tiny is the selected primary workload candidate because it meets the
project's target range (about 5.7M parameters and 1.3 GMAC/inference) and its
projection layers provide a natural workload for Xqmac. Selection is
conditional on producing and validating a W8A8 implementation; the available
LiteRT artifact is FLOAT32 and is not itself suitable for the bare-metal target.

The first deliverable will be a quantized, host-verified DeiT Transformer block
with static weights and buffers. Full-model inference remains the goal, but is
not yet guaranteed: the feasibility gates are W8A8 conversion/accuracy,
integer support for the graph's operators, and target memory capacity. If these
gates fail, keep the complete-block plus two-major-operator deliverable and
document the measured reason full inference was infeasible. Do not silently
replace DeiT with the existing synthetic GEMM benchmark.

The pinned artifact, FlatBuffer inspection results, and next feasibility check
are recorded in `models/deit_tiny_patch16_224/README.md`.

The first model-shaped bare-metal workload is a Block 0 MLP `fc1` tile
(`deit-block0-mlp-fc1-tile-v1`): real W8A8 pilot weights, deterministic INT8
activations, host INT32 golden, and B0/B1/X1 Spike measurements documented in
`EXPERIMENTS.md`. Full-operator and full-block ports remain follow-on work.
Requantization is not yet part of the X1 implementation despite the eventual
X1 requirement.

Model files and conversion outputs are inputs to the experiment. The processor
work is the bare-metal kernels, ISA extension, simulator, and RTL changes that
execute those inputs.

## Xqmac8 Semantics

The initial Xqmac8 proposal uses two packed 64-bit source operands and one
64-bit accumulator input:

```text
rd = rs3 + sum(signed(rs1[8*i +: 8]) * signed(rs2[8*i +: 8])) for i = 0..7
```

The byte at bits 7:0 is lane 0, and the byte at bits 63:56 is lane 7. Every
lane is interpreted as signed INT8. The eight products and the incoming
accumulator are sign-extended to INT64 before addition. The initial design
does not saturate and does not use floating point; signed INT64 wraparound is
the architectural overflow behavior to be confirmed against the chosen RTL
implementation.

The software oracle is `src/xqmac8.c`, tested by `tests/test_xqmac8.c`. The
custom opcode encoding and Spike decoder integration are implemented in
`src/xqmac8_encoding.h` and `src/xqmac8_spike.cc`. The X1 GEMM kernel
(`int8_gemm_x1`) accepts a general `inner` dimension that is a multiple of 8
and issues `inner/8` Xqmac8 operations per output element.

## Scalar Requantization Semantics

Software requantization (pre-ISA) is implemented in `src/requantize.c`:

```text
tmp = (acc * multiplier + (1 << (shift - 1))) >> shift
out = sat_int8(tmp + zero_point)
```

- `acc` is a signed INT32 GEMM accumulator
- `multiplier` is a signed INT32 fixed-point factor
- `shift` is a non-negative arithmetic right shift
- `zero_point` is added after the shift
- saturation clamps to `[-128, 127]`
- no floating-point operations

The DeiT fc1 tile uses per-channel multipliers with a shared shift, generated
from pilot scales by `scripts/generate_fc1_requant_params.py`. B0/B1 keep this
scalar path. X1 replaces it with the custom `xrequant` instruction below.

## Chained MLP Workload Design

The Block 0 MLP chain (`src/deit_mlp_b0_main.c`, `src/deit_mlp_x1_main.c`)
is the first workload where operator outputs feed operator inputs:

```text
x[192] -> fc1 -> +bias_eff -> requant -> GELU LUT -> fc2 -> +bias_eff -> requant -> y[192]
```

Design decisions:

- **Asymmetric activations are handled offline.** TFLite computes
  `sum (x - zp_in) * w + bias`. The extractor folds the zero-point into the
  bias (`bias_eff = bias - zp_in * sum(w)`), so the on-target GEMM stays a
  plain INT8 dot product and Xqmac8 needs no zero-point support.
- **GELU is a 256-entry INT8->INT8 LUT**, generated offline, one load per
  element. No floating point on the target and no new instruction for it.
- **Per-channel requant, shared shift.** Multipliers are per output channel;
  the right-shift is common to a layer and capped at 31 so that
  `acc * multiplier` fits in 64 bits and the shift fits `xrequant`'s 8-bit
  field.
- **No repacking between layers on X1.** The INT8 activation vector is stored
  in an 8-byte-aligned `uint64_t` buffer, so its lanes already form Xqmac8
  operands; TFLite's `[N, K]` weight layout is already per-channel packed.
- **Per-stage functions are `noinline`** so Spike traces can be attributed to
  GEMM, fc1 requant, GELU and fc2 requant (`scripts/spike_metrics.py`).

Tensor indices used by the extractors were checked against the model's op graph
(`scripts/probe_mlp_ops.py`): fc1 184 -> 185, GELU 185 -> 186, fc2 186 -> 187
(not 188, which is the residual ADD output).

## Xrequant Semantics

```text
xrequant rd, rs1, rs2

ctrl = rd
  zero_point = signed(ctrl[7:0])
  shift      = unsigned(ctrl[15:8])
acc  = signed32(rs1)
mult = signed32(rs2)
tmp  = (acc * mult + (1 << (shift-1))) >> shift   # if shift > 0
rd   = sext_int8(sat_int8(tmp + zero_point))
```

Encoding (custom-0 family, distinct from Xqmac8 by `funct3`):

- opcode = `0x0b`, funct3 = `0x1`, funct7 = `0x2b`
- `xrequant a0, a1, a2` encodes as `0x56c5950b`

Software helpers: `src/xrequant_encoding.h`, `src/xrequant.h`.  
Spike decode/execute lives beside Xqmac8 in `src/xqmac8_spike.cc` and is loaded
with `--extension=xqmac8`.
