## Reproducibility Record

Use one row for every benchmark/configuration pair. Keep the input data and
matrix shapes unchanged when comparing configurations.

| Configuration | Workload/operator | Input ID | Toolchain | Simulator/RTL | Result file | Status |
|---|---|---|---|---|---|---|
| B0 | Synthetic 8x16x8 INT8 GEMM | synthetic-v1 | RISC-V GCC 13.2, `-O0` | Spike 1.1.1-dev | `build/B0/trace.log` | Complete |
| B1 | Synthetic 8x16x8 INT8 GEMM | synthetic-v1 | RISC-V GCC 13.2, `-O2` | Spike 1.1.1-dev | `build/B1/trace.log` | Complete |
| X1 | Synthetic 8x16x8 INT8 GEMM | synthetic-v1 | RISC-V GCC 13.2, `-O2` + Xqmac8 | Spike 1.1.1-dev | `build/X1/trace.log` | Complete |
| B0 | DeiT Block0 MLP fc1 tile 8x192x8 | deit-block0-mlp-fc1-tile-v1 | RISC-V GCC 13.2, `-O0` | Spike 1.1.1-dev | `build/B0_deit_fc1/trace.log` | Complete |
| B1 | DeiT Block0 MLP fc1 tile 8x192x8 | deit-block0-mlp-fc1-tile-v1 | RISC-V GCC 13.2, `-O2` | Spike 1.1.1-dev | `build/B1_deit_fc1/trace.log` | Complete |
| X1 | DeiT Block0 MLP fc1 tile 8x192x8 | deit-block0-mlp-fc1-tile-v1 | RISC-V GCC 13.2, `-O2` + Xqmac8 | Spike 1.1.1-dev | `build/X1_deit_fc1/trace.log` | Complete |
| B0 | DeiT fc1 tile + scalar requant | deit-block0-mlp-fc1-tile-requant-v1 | RISC-V GCC 13.2, `-O0` | Spike 1.1.1-dev | `build/B0_deit_fc1_requant/trace.log` | Complete |
| B1 | DeiT fc1 tile + scalar requant | deit-block0-mlp-fc1-tile-requant-v1 | RISC-V GCC 13.2, `-O2` | Spike 1.1.1-dev | `build/B1_deit_fc1_requant/trace.log` | Complete |
| X1 | DeiT fc1 tile Xqmac8 + scalar requant | deit-block0-mlp-fc1-tile-requant-v1 | RISC-V GCC 13.2, `-O2` + Xqmac8 | Spike 1.1.1-dev | `build/X1_deit_fc1_requant/trace.log` | Complete |
| B0/B1/X1 | DeiT Block0 attention QKV tile 8x192x8 + requant | deit-block0-qkv-tile-v1 | RISC-V GCC 13.2, `-O0` / `-O2` / `-O2` + Xqmac8 + Xrequant | Spike 1.1.1-dev | `build/{B0,B1,X1}_op_qkv/trace.log`, `results/deit_qkv_metrics.md` | Complete |
| B0/B1/X1 | DeiT Block0 MLP fc2 tile 8x768x8 + requant | deit-block0-fc2-tile-v1 | RISC-V GCC 13.2, `-O0` / `-O2` / `-O2` + Xqmac8 + Xrequant | Spike 1.1.1-dev | `build/{B0,B1,X1}_op_fc2/trace.log`, `results/deit_fc2_metrics.md` | Complete |
| B0/B1/X1 | DeiT Block0 MLP chain, 1 token: fc1 -> requant -> GELU LUT -> fc2 -> requant | deit-block0-mlp-chain-token0-v1 | RISC-V GCC 13.2, `-O0` / `-O2` / `-O2` + Xqmac8 + Xrequant | Spike 1.1.1-dev | `results/deit_mlp_metrics.md` | Complete |
| X2 | TBD | TBD | TBD | TBD | TBD | Optional |

Record at minimum:

- correctness result against the host reference;
- dynamic instructions and instruction mix;
- dynamic memory instructions and MACs per instruction;
- code size;
- cycle count and CPI when measured on CV-Wally;
- compiler flags, simulator settings, and source revision.

## Rerun Procedure

When a benchmark is missing, rerun the named configuration with its original
input-data identifier and settings. Do not use the current default implicitly.
The expected command interface is:

```text
run-benchmark --config B0 --input <input-id>
run-benchmark --config B1 --input <input-id>
run-benchmark --config X1 --input <input-id>
```

The commands are a planned interface until the toolchain and benchmark runner
are added.

## B0 Initial Spike Smoke Test

The first finite B0 run uses the hand-written 2 x 3 by 3 x 2 INT8 GEMM test.
It is a toolchain and trace-validation workload, not yet a representative
ResNet-18 operator.

Command:

```text
spike --isa=RV64IMC -l build/B0/b0.elf > build/B0/trace.log 2>&1
```

```text
Full command was:

 cd /mnt/c/Users/ryang/OneDrive/Documents/Advanced_Computer_Architecture_course_project; echo RETIRED=$(grep-c '^core' build/B0/trace.log); echo MULW=$(grep '^core' build/B0/trace.log | grep -E '\) mulw ' | wc -l); echo LOADS=$(grep '^core' build/B0/trace.log | grep -E '\) (lb|lbu|lh|lhu|lw|ld) ' | wc -l); echo STORES=$(grep '^core' build/B0/trace.log | grep -E '\) (sb|sh|sw|sd) ' | wc -l); echo BRANCHES=$(grep '^core' build/B0/trace.log | grep -E '\) (beq|bne|blt|bge|bltu|bgeu|jal|jalr|j) ' | wc -l); echo COMPRESSED=$(grep '^core' build/B0/trace.log | grep -E '\) c\.' | wc -l)
```

Observed result:

| Metric | Value |
|---|---:|
| Spike exit code | 0 |
| Retired trace lines | 5,003 |
| MAC operations | 12 |
| `mulw` instructions | 40 |
| Load instructions | 246 |
| Store instructions | 72 |
| Control-flow instructions | 31 |
| Compressed instructions | 4,587 |
| Compiler optimization | `-O0` |

The result confirms that the bare-metal program executes correctly in Spike and
that the trace parser sees scalar multiply and memory instructions. The final
performance baseline must use larger GEMM dimensions and extracted ResNet-18
operators, with identical inputs across B0, B1, and X1.

## B0 Synthetic GEMM Baseline

The first meaningful baseline uses an 8 x 16 activation matrix multiplied by a
16 x 8 weight matrix. The input values are deterministic static INT8 arrays in
`src/b0_main.c`; the output is accumulated in INT32. This workload performs
1,024 MAC operations and is intended for comparing B0 against B1 and X1.

Command:

```text
/home/rachelyzy/riscv/bin/spike --isa=RV64IMC -l build/B0/b0.elf > build/B0/trace.log 2>&1
```

Observed result:

| Metric | Value |
|---|---:|
| Spike exit code | 0 |
| Retired trace lines | 45,003 |
| MAC operations | 1,024 |
| `mulw` instructions | 3,136 |
| Load instructions | 17,069 |
| Store instructions | 4,312 |
| Control-flow instructions | 1,175 |
| Compressed instructions | 19,282 |
| Compiler optimization | `-O0` |

This is still a synthetic workload, not an extracted ResNet-18 operator. It
provides a stable input and measurement contract while the baseline and custom
instruction flow is developed.

## Xqmac8 Reference Semantics

The host-side Xqmac8 oracle and directed test pass for packed signed INT8 lanes,
including negative values, `127`, `-128`, and a nonzero accumulator. The test
command is:

```text
gcc -std=c11 -Wall -Wextra -Werror -Isrc src/xqmac8.c tests/test_xqmac8.c -o /tmp/test_xqmac8
/tmp/test_xqmac8
```

Observed result:

```text
Xqmac8 reference test passed: -12107
```

This validates the software semantics only. It does not yet validate a custom
Spike instruction, assembler encoding, CV-Wally RTL, or performance.

## B1 Software-Optimized Synthetic GEMM

B1 uses the identical source workload and input data as B0. The only intended
change is compiler optimization: `-O2` instead of `-O0`. The binary is built by
`scripts/build_b1.sh` and stored under `build/B1/`.

Observed result:

| Metric | B0 | B1 |
|---|---:|---:|
| Spike exit code | 0 | 0 |
| Retired trace lines | 45,003 | 7,830 |
| MAC operations | 1,024 | 1,024 |
| `mulw` instructions | 3,136 | 1,024 |
| Load instructions | 17,069 | 2,050 |
| Store instructions | 4,312 | 65 |
| Control-flow instructions | 1,175 | 1,175 |
| Compressed instructions | 19,282 | 3,326 |
| Text bytes | 724 | 514 |
| Data bytes | 72 | 72 |
| BSS bytes | 4,360 | 4,360 |

B1 reduces dynamic instructions by approximately 82.6% and scalar multiply
instructions by approximately 67.3% for this synthetic workload. These are
Spike instruction-count improvements, not cycle-level claims. CV-Wally or a
documented cycle model is required for cycle speedup.

## Isolated GEMM Kernel Profile

The reusable analyzer `scripts/analyze_kernel_trace.sh` filters each Spike trace
to the address range of `int8_gemm_reference`. This separates the targeted
kernel from startup, output checking, and the `tohost` completion sequence.

Commands:

```text
bash scripts/analyze_kernel_trace.sh build/B0/trace.log 800000ae 800001d4 1024
bash scripts/analyze_kernel_trace.sh build/B1/trace.log 8000006e 80000102 1024
```

Observed kernel-only result:

| Metric | B0 | B1 |
|---|---:|---:|
| GEMM instructions | 42,084 | 7,788 |
| MAC operations | 1,024 | 1,024 |
| `mulw` instructions | 3,136 | 1,024 |
| Loads | 17,066 | 2,048 |
| Stores | 4,311 | 64 |
| Control-flow instructions | 1,169 | 1,169 |
| MACs per instruction | 0.024332 | 0.131484 |

The isolated GEMM instruction count falls by approximately 81.5% from B0 to B1.
This is the targeted comparison for future ISA extensions; full-program traces
remain necessary for end-to-end results.

## X1 Xqmac8 Synthetic GEMM

X1 uses the same deterministic 8 x 16 by 16 x 8 workload as B0 and B1. The
packed kernel stores eight signed INT8 values in each 64-bit operand and uses
two Xqmac8 instructions per output value. The 64 outputs therefore execute
128 Xqmac8 instructions.

The Spike extension was loaded with the patched custom-extension build, and
the benchmark passed its `tohost` correctness check.

Observed result, counted through the first completion loop:

| Metric | B1 | X1 |
|---|---:|---:|
| Spike exit code | 0 | 0 |
| Committed instructions | 7,826 | 897 |
| Xqmac8 instructions | 0 | 128 |
| Text bytes | 514 | 470 |
| Data bytes | 72 | 72 |
| BSS bytes | 4,360 | 4,360 |

X1 reduces the bounded dynamic instruction count by approximately 88.5%
relative to B1 for this synthetic workload. These are ISA-level Spike counts;
cycle speedup and CPI still require CV-Wally or a documented cycle model.

## DeiT Block0 MLP fc1 Tile (Model-Shaped Workload)

This workload ports a real operator tile from the W8A8 DeiT-Tiny pilot
(`deit_tiny_w8a8_pilot.tflite`) into the existing B0/B1/X1 bare-metal path.

Metadata: `models/deit_tiny_patch16_224/deit_fc1_tile_meta.json`  
Input ID: `deit-block0-mlp-fc1-tile-v1`  
Shape: 8 x 192 activations by 192 x 8 weight columns (12,288 MACs)  
Weights: first 8 output channels of Block 0 `mlp.fc1` (TFLite `[768,192]`,
transposed to `[K,N]` for `int8_gemm_reference`)  
Activations: deterministic INT8 values (`numpy` RNG seed `20260929`, range
`[-8, 8]`) so the tile is reproducible without intermediate-tensor capture  
Generation: `scripts/extract_deit_gemm_tile.py`  
Builds: `scripts/build_{b0,b1,x1}_deit_fc1.sh`  
Runner: `scripts/run_deit_fc1_configs.sh`

Spike invocation:

```text
spike --isa=RV64IMC -l build/B0_deit_fc1/b0_deit_fc1.elf
spike --isa=RV64IMC -l build/B1_deit_fc1/b1_deit_fc1.elf
LD_LIBRARY_PATH=$HOME/riscv:$HOME/riscv/lib \
  spike --isa=RV64IMC --extension=xqmac8 -l build/X1_deit_fc1/x1_deit_fc1.elf
```

Correctness: all three configurations exit Spike with code 0 (`tohost` pass)
against the host-generated INT32 golden array embedded in
`src/deit_fc1_tile_data.h`.

### Full-program counts (bounded before spin)

Produced by `scripts/summarize_deit_fc1_metrics.py`:

| config | retired | macs | mulw | xqmac8 | loads | stores | control_flow | macs_per_instr | loads_per_mac |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 481407 | 12288 | 36928 | 0 | 197291 | 49432 | 12509 | 0.025525 | 16.055583 |
| B1 | 87054 | 12288 | 12288 | 0 | 24705 | 65 | 12566 | 0.141154 | 2.010498 |
| X1 | 16555 | 12288 | 0 | 1536 | 3201 | 65 | 1814 | 0.742253 | 0.260498 |

| Comparison | Dynamic instruction reduction |
|---|---:|
| B1 vs B0 | 81.92% (~5.53× fewer) |
| X1 vs B1 | 80.98% (~5.26× fewer) |
| X1 vs B0 | 96.56% (~29.1× fewer) |

Code size from `riscv64-unknown-elf-size` (`.text` column includes embedded
rodata for these freestanding links):

| config | text | data | bss |
|---|---:|---:|---:|
| B0 | 3792 | 72 | 4360 |
| B1 | 3588 | 72 | 4360 |
| X1 | 3624 | 72 | 4360 |

### Isolated GEMM kernel profile

Commands:

```text
bash scripts/analyze_kernel_trace.sh build/B0_deit_fc1/trace.log 800000aa 800001d0 12288
bash scripts/analyze_kernel_trace.sh build/B1_deit_fc1/trace.log 80000070 80000104 12288
bash scripts/analyze_kernel_trace.sh build/X1_deit_fc1/trace.log 80000070 80000128 12288
```

| Metric | B0 | B1 | X1 |
|---|---:|---:|---:|
| GEMM instructions | 481380 | 86636 | 13065 |
| MAC operations | 12288 | 12288 | 12288 |
| `mulw` | 36928 | 12288 | 0 |
| `xqmac8` | 0 | 0 | 1536 |
| Loads (incl. compressed) | 197290 | 24576 | 3072 |
| Stores (incl. compressed) | 49431 | 64 | 64 |
| Control-flow | 12506 | 12433 | 1681 |
| MACs per instruction | 0.025527 | 0.141835 | 0.940528 |

Kernel-only X1 uses exactly `rows * columns * (inner/8) = 8 * 8 * 24 = 1536`
Xqmac8 instructions and 3072 packed loads (two `c.ld` per Xqmac8). Relative to
B1, the isolated kernel instruction count falls by approximately 84.9%
(~6.63×), which satisfies the project target of ≥2× dynamic-instruction
reduction on a major GEMM/FC kernel.

These remain ISA-level Spike counts. CV-Wally cycle/CPI/area measurements and
a custom requantization *instruction* are still outstanding. The W8A8 source
model is still the one-image pilot; replace with a calibrated export before
claiming accuracy-qualified full-model results.

## DeiT fc1 Tile + Scalar Requantization

After the INT32 GEMM tile, each of the 64 outputs is requantized to INT8 with
the software kernel in `src/requantize.c`:

```text
tmp = (acc * multiplier[channel] + (1 << (shift-1))) >> shift
out = sat_int8(tmp + zero_point)
```

Parameters are derived from the W8A8 pilot (`scripts/generate_fc1_requant_params.py`,
metadata `deit_fc1_requant_meta.json`):

- activation scale from tensor 184
- per-channel weight scales from tensor 69 (first 8 channels)
- output scale/ZP from tensor 185 (`scale≈0.0638`, `zp=7`)
- shared `shift=31`, 8 channel multipliers

This path does **not** yet subtract activation zero-points or add the INT32
bias; it is a scalar requant microbenchmark on the existing GEMM tile outputs.

Host test: `tests/test_requantize.c` passes. Spike exit codes: B0/B1/X1 = 0.

Bounded full-program counts for GEMM + scalar requant:

| config | retired | mulw | xqmac8 | loads | stores | control_flow |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 481465 | 36929 | 0 | 197303 | 49440 | 12515 |
| B1 | 88798 | 12352 | 0 | 24833 | 129 | 12894 |
| X1 | 18299 | 64 | 1536 | 3329 | 129 | 2142 |

Compared with the prior B1/X1 GEMM-only tile (`retired` 87054 / 16555), adding
scalar requant costs about **+1744 dynamic instructions** for 64 outputs
(~**27.3 instructions/output element**). On X1, GEMM is already packed/Xqmac8,
so requantization is a visible remaining software bottleneck.

## X1 with Xrequant Custom Instruction

X1 now pairs Xqmac8 GEMM with the custom `xrequant` instruction (scalar
requant remains on B0/B1). Rebuild Spike via `scripts/install_xqmac8_spike.sh`
and run `scripts/run_xrequant_fc1.sh`.

Correctness: Spike exit 0 for B1 (scalar) and X1 (`xrequant`), matching the
same INT8 golden array. Trace shows exactly **64** `xrequant` and **1536**
`xqmac8` on X1.

Bounded full-program counts:

| config | retired | mulw | xqmac8 | xrequant | loads | stores | control_flow |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 scalar requant | 88798 | 12352 | 0 | 0 | 24833 | 129 | 12894 |
| X1 xqmac8 + xrequant | 17204 | 0 | 1536 | 64 | 3329 | 129 | 1886 |

X1 vs B1: **80.63%** fewer dynamic instructions (~5.16×). Relative to the prior
X1 path that used **scalar** requant (`retired` 18299), `xrequant` removes about
**1095** instructions (~**17.1 instructions/output** saved) while collapsing the
requant multiply/round/shift/ZP/sat sequence into one retired instruction per
output element.

## Operator Coverage: fc1, QKV, fc2 (Generic Pipeline)

Three Block 0 operators run through one generic path
(`scripts/run_deit_op.sh <fc1|qkv|fc2>`): `extract_deit_op.py` generates
`src/deit_<op>_data.h`, `build_deit_op.sh` builds B0 (`-O0`, scalar GEMM +
scalar requant), B1 (`-O2`, same sources) and X1 (`-O2`, Xqmac8 GEMM +
`xrequant`), and `spike_metrics.py` produces the tables in
`results/deit_<op>_metrics.md`. All configs of all three operators exit 0
(INT32 GEMM output and INT8 requant output both match host golden arrays).

Each tile is 8 rows x K x 8 output channels followed by per-channel requant of
64 outputs, with scales from the W8A8 pilot (one-image calibration; accuracy
not qualified). Counts cover retired instructions at `pc >= 0x80000000` with
the trailing `tohost` spin removed, identically for every config.

| Operator | Tile (MxKxN) | MACs | B0 retired | B1 retired | X1 retired | B1 vs B0 | X1 vs B1 | X1 vs B0 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fc1 (192->768) | 8x192x8 | 12,288 | 489,010 | 89,050 | 14,446 | 5.49x | 6.16x | 33.85x |
| qkv (192->576) | 8x192x8 | 12,288 | 489,010 | 89,050 | 14,446 | 5.49x | 6.16x | 33.85x |
| fc2 (768->192) | 8x768x8 | 49,152 | 1,926,706 | 347,098 | 51,310 | 5.55x | 6.76x | 37.55x |

X1 detail (MACs per retired instruction and loads+stores per MAC):

| Operator | B0 MAC/instr | B1 MAC/instr | X1 MAC/instr | B1 (ld+st)/MAC | X1 (ld+st)/MAC |
|---|---:|---:|---:|---:|---:|
| fc1 / qkv | 0.0251 | 0.1380 | 0.8506 | 2.0417 | 0.2917 |
| fc2 | 0.0255 | 0.1416 | 0.9579 | 2.0104 | 0.2604 |

Observations:

- fc1 and qkv have **identical instruction counts** because the tiles share a
  shape and the kernels' control flow does not depend on data values. QKV adds
  operator/scale diversity (attention vs MLP, different weights and requant
  parameters) but not a different performance profile. fc2 is the operator
  with a genuinely different profile (4x longer reduction).
- The Xqmac8 GEMM instruction count scales exactly with `rows * columns *
  K/8`: 1,536 for K=192 and 6,144 for K=768.
- B1's gain over B0 is about the same on fc2 (5.55x) as on fc1 (5.49x), and
  X1's gain over B1 grows slightly with K (6.76x vs 6.16x) as the fixed
  per-output requant and `main` overhead becomes a smaller share.
- **Correction (2026-10-15):** an earlier revision of this file reported B0
  fc2 as 1,198,428 instructions (B1 vs B0 3.45x, X1 vs B0 23.36x) and drew a
  conclusion that B1's gain shrinks with K. That B0 trace was written to the
  OneDrive mount and was evidently incomplete (about 24 instructions/MAC
  versus ~39 for every other B0 run). Traces are now written to the Linux
  filesystem and the rerun gives 1,926,706. The earlier conclusion was an
  artifact and is withdrawn.
- **Correction (2026-10-15):** the first fc2 run used tensor 188 (the residual
  ADD output) as the requant output; the fc2 FULLY_CONNECTED output is tensor
  187 (scale 0.0329, zp 26). `extract_deit_op.py` now uses 187. B1/X1 counts
  did not change; golden outputs and requant parameters did.
- Per-function split on X1 fc2: `int8_gemm_x1` retires 49,929 of 51,310
  instructions (97%). The requant instructions are 64 of them, so for large K
  the packed GEMM loop (pointer arithmetic, two loads per Xqmac8, loop
  branch) dominates, not requantization.
- Method note: these generic-pipeline counts supersede the earlier fc1
  bounded counts in this file, which used a slightly different bounding
  heuristic (differences of a few instructions to ~1%).

## Chained Workload: Block 0 MLP for One Token (fc1 -> GELU -> fc2)

Unlike the isolated tiles above, this program chains operators so each output
is the next operator's input, at **full layer width** for **one token**:

```text
x[192] -> fc1 (192->768) -> +bias -> requant -> GELU (256-entry LUT)
       -> fc2 (768->192) -> +bias -> requant -> y[192]
```

Input id `deit-block0-mlp-chain-token0-v1`; metadata and accuracy numbers in
`models/deit_tiny_patch16_224/deit_mlp_chain_meta.json`; metrics in
`results/deit_mlp_metrics.md`. Run: `bash scripts/run_deit_op.sh mlp`.

How it is built:

- **Real activations.** `sample.png` is run through the INT8 TFLite model on
  the host (no delegates, all tensors preserved). The INT8 activation entering
  fc1 (tensor 184, token 0 = CLS) is the chain input. Preprocessing is a plain
  224x224 bicubic resize with ImageNet mean/std, not timm's exact pipeline;
  this is fine for an architecture workload but not for task accuracy.
- **Zero-points and bias folded offline.** `acc = sum x*w + bias_eff`, with
  `bias_eff = bias - zp_in * sum(w)`, so the target needs one add per output
  and no per-element zero-point math. This fixes a limitation of the earlier
  tiles, which ignored activation zero-points and bias.
- **GELU as a LUT.** 256-entry INT8->INT8 table generated offline from the
  exact erf GELU using the model's input/output scales. No floating point on
  the target. 0 entries needed patching to match TFLite's observed behaviour.
- **Per-channel requant** for all 768 and 192 output channels (shared shift).
- **X1 feeds fc2 with no repacking.** The GELU output lives in a
  `uint64_t`-backed buffer, so eight INT8 lanes are already one Xqmac8
  operand; TFLite's `[N, K]` weight layout is already the per-channel packed
  layout, so packed weights are used as stored.

Integer parity of the host chain against the model's own tensors (all 197
tokens, so this is a broader check than the single token run on target):

| Stage | Exact match | Max abs diff |
|---|---:|---:|
| fc1 output vs TFLite | 99.82% | 1 |
| GELU output vs TFLite | 99.91% | 3 |
| fc2 output vs TFLite, given TFLite's GELU output | 99.91% | 1 |
| fc2 output vs TFLite, end to end | 97.62% | 1 |

All mismatches are small (max 1 LSB, except GELU at 3). The likely causes are
rounding-mode differences (TFLite rounds ties away from zero; this chain
rounds half up) and the shared-shift multiplier approximation, with a +-1
difference in fc1 amplified through GELU's slope; I have not isolated these
individually. This validates integer parity with the pilot model, not
classification accuracy: the pilot was calibrated on a single image.

Spike results (all three configs exit 0, matching golden GELU output and final
output; all counts are for one token):

| config | retired | MACs/instr | (ld+st)/MAC | instr/output | multiply | xqmac8 | xrequant | load | store | branch/jump |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 11,651,515 | 0.0253 | 20.19 | 12,137 | 886,656 | 0 | 0 | 4,758,740 | 1,195,609 | 306,273 |
| B1 | 2,111,029 | 0.1397 | 2.03 | 2,199 | 295,872 | 0 | 0 | 596,160 | 2,689 | 305,301 |
| X1 | 328,109 | 0.8988 | 0.28 | 342 | 0 | 36,864 | 960 | 80,064 | 2,689 | 42,453 |

| Comparison | Reduction |
|---|---:|
| B1 vs B0 | 81.88% (5.52x) |
| X1 vs B1 | 84.46% (6.43x) |
| X1 vs B0 | 97.18% (35.51x) |

Per-function retired instructions (the operator contribution breakdown):

| function | B0 | B1 | X1 | share of X1 |
|---|---:|---:|---:|---:|
| GEMM (`int8_gemm_reference` / `int8_gemm_x1`, fc1 + fc2) | 11,531,418 | 2,072,140 | 304,604 | 92.8% |
| `stage_requant_fc1` (+ `requantize_i32_to_i8` on B0/B1) | 29,200 (+46,080) | 10,009 (+14,400) | 9,229 | 2.8% |
| `stage_gelu` (LUT) | 19,214 | 6,152 | 6,152 | 1.9% |
| `stage_requant_fc2` | 7,312 | 2,521 | 2,317 | 0.7% |
| `main` + `_start` | 18,291 | 5,807 | 5,807 | 1.8% |

(On B0/B1 the shared `requantize_i32_to_i8` function is reported separately
and covers both requant stages: 46,080 and 14,400 instructions. On X1 the
requant is the inline `xrequant` instruction, so it is already inside the
`stage_*` rows.)

Findings:

- Chaining and bias/zero-point/LUT handling add real overhead that the
  isolated tiles hid, but the ordering of configurations is unchanged and the
  speedups are consistent with the tile results (5.5x / 6.4x).
- On X1 the GEMM is still 92.8% of all retired instructions, so after Xqmac8
  and `xrequant` the next bottleneck is the packed-GEMM loop itself:
  per Xqmac8 there are two 8-byte loads plus pointer increments and a loop
  branch. That is 36,864 Xqmac8 versus 304,604 GEMM instructions, i.e. about
  8.3 instructions per Xqmac8, of which only one does MAC work. Reducing
  load/address/loop overhead is therefore the profile-justified direction for
  X2 (e.g. post-increment loads or hardware loops).
- Requantization is no longer significant here (3.5% of X1), partly because
  this is one token; for a full 197-token sequence both GEMM and requant
  counts scale linearly with token count, so the shares stay about the same.
  Whole-MLP estimate for 197 tokens: roughly 197x the per-token counts
  (X1 about 64.6M instructions), which is an extrapolation, not a
  measurement.

Not yet done: attention (Q x K^T, softmax, x V, output projection), LayerNorm
and residual adds, so a complete Transformer block is still outstanding;
calibrated W8A8 with accuracy validation; and cycle/CPI numbers on CV-Wally
RTL.
