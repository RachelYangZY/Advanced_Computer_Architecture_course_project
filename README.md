# Quantized DeiT-Tiny on bare-metal RV64IMC with packed INT8 MAC and requantization instructions

Term project for Advanced Computer Architecture. We accelerate W8A8 DeiT-Tiny inference on an RV64IMC core
(CV-Wally) with two custom instructions and evaluate them first on Spike (ISA level), then in CV-Wally RTL.

| Instruction | What it does |
|---|---|
| `xqmac8 rd, rs1, rs2` | `rd += sum(int8(rs1[i]) * int8(rs2[i]))` for 8 packed INT8 lanes: 8 MACs per instruction. `rd` is both accumulator input and result. |
| `xrequant rd, rs1, rs2` | Fused TFLite-exact requantization: saturating rounding doubling high multiply, rounding right shift, zero-point add, INT8 saturation. `rd` supplies a control word (zero point, shift, mode). |

Both live in the RISC-V custom-0 opcode space (`0x0b`, `funct7 = 0x2b`, `funct3` 0 / 1). Semantics and encoding:
[`DESIGN.md`](DESIGN.md).

## Contents

1. [Status](#status)
2. [Results](#results)
3. [Workflow](#workflow)
4. [Setup](#setup)
5. [Running everything](#running-everything)
6. [Tests and verification](#tests-and-verification)
7. [Configurations](#configurations)
8. [Limitations](#limitations)
9. [Repository layout](#repository-layout)

## Status

| Stage | State |
|---|---|
| Model: DeiT-Tiny W8A8 (5.7 M params, 1.25 GMAC), converted to integer-only C by `tools/tflite2c` | Done, bit-exact vs TFLite on 3 inputs |
| B0, B1, X1, X1MAC kernel libraries, full inference on Spike | Done |
| `xqmac8`, `xrequant`: semantics, Spike implementation, directed + randomized verification | Done |
| ISA-level metrics: instruction mix, memory instructions, MAC/instr, code size, custom-instruction use | Done |
| CV-Wally RTL, synthesis (area, Fmax), cycle/CPI measurements | Planned |
| X2 (extra extension chosen after profiling X1) | Not started (optional) |

## Results

Whole model, one inference of `sample.png`, 1,253,683,200 MACs, counted on Spike (`minstret`). All four
configurations produce all 1000 logits identical to TFLite. Source: `results/model/profile.md`,
`results/model/instr_mix.md`.

| Config | Instructions | vs B0 | vs B1 | Instr / MAC | MAC / instr | (Load+store) / MAC | `.text` bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 12,598,660,515 | 1.00x | | 10.05 | 0.10 | 2.22 | 31,786 |
| B1 | 6,478,403,747 | 1.94x | 1.00x | 5.17 | 0.19 | 0.94 | 34,192 |
| **X1** | **1,522,226,359** | **8.28x** | **4.26x** | **1.22** | **0.82** | **0.28** | 34,214 |
| X1MAC | 1,580,552,607 | 7.97x | 4.10x | 1.26 | 0.79 | 0.28 | 34,416 |

- **Kernel level:** the shared GEMM core (`gemm_nt_rows`) goes from 5.39 G instructions (B1) to 0.49 G (X1), 11.0x.
  All fully-connected ops: 8.60 G (B0) to 0.56 G (X1), 15.4x.
- **Contribution of each instruction:** `xrequant` adds 3.7% on top of `xqmac8` (X1MAC to X1). Almost all of the gain is `xqmac8`.
- **Custom-instruction use in X1:** 157.7 M `xqmac8` (10.3% of instructions), 6.0 M `xrequant` (0.4%).
- **Multiplies:** 10.4% of B0 instructions and 19.8% of B1, but 1.1% of X1 (1.04 to 0.014 per MAC).
- **Next bottleneck:** after X1, non-GEMM ops (ADD, MUL, SQUARED_DIFFERENCE, transpose, softmax, mean, ...) are 52% of
  instructions. GEMM ops fall from 94% (B0) to 48% (X1).

Project targets: at least 2x fewer instructions in a major GEMM/FC kernel (**11.0x**), at least 1.5x end to end
(**8.28x**, in instruction count), output matches the reference (**bit-exact**).

**These are instruction counts, not time.** Spike reports `mcycle == minstret`; cycles and CPI need CV-Wally.
Details, per-op and per-block tables, and the three largest bottlenecks for each configuration are in
[`EXPERIMENTS.md`](EXPERIMENTS.md) and `results/model/`. Figures: `docs/figures/` (`instr_breakdown.png`, `model_map.png`).

## Workflow

```text
.tflite --(tflite_ir.py, gen_model.py)--> bare-metal C --(build_model.sh)--> model.elf --(Spike)--> counts + logits
  |                                         weights.c, model.c,        one ELF per config      xqmac8 plugin loaded
  |                                         model_io.c (golden I/O)    B0 / B1 / X1 / X1MAC    with --extension=xqmac8
  +--> TFLite interpreter --> golden tensors t*.bin
```

Three independent checks:

| # | Check | Script |
|---|---|---|
| 1 | Generated C equals the TFLite interpreter on every tensor (host build) | `gen_check.sh` |
| 2 | On Spike, all 1000 output logits equal the embedded TFLite golden values | `run_model.sh` |
| 3 | Every executed custom instruction equals its C oracle (directed and random) | `verify_isa.sh` |

B0, B1 and X1 share the same generated `model.c`. Only the kernel library linked at build time changes
(`scripts/build/build_model.sh`):

| Config | Kernel sources | Compile flags |
|---|---|---|
| B0 | `src/gen/ops_ref.c` | none |
| B1 | `ops_ref.c` + `src/gen/ops_opt.c` | none |
| X1 | `ops_ref.c` + `ops_opt.c` | `-DGEMM_XQMAC8 -DUSE_XREQUANT` |
| X1MAC | `ops_ref.c` + `ops_opt.c` | `-DGEMM_XQMAC8` |

X1 programs are run with `spike --isa=RV64IMC --extension=xqmac8 <elf>`.
Generator details: [`docs/GENERATOR.md`](docs/GENERATOR.md).

## Setup

Scripts are written for **WSL Ubuntu**. Needed:

- `riscv64-unknown-elf-gcc` (13.2 was used) and binutils on `PATH`
- Spike (1.1.1-dev was used) installed at `~/riscv/bin/spike`, with its source tree at `~/riscv-isa-sim`
  (the extension installer copies the plugin into that tree and builds it)
- Python 3 and `virtualenv`; host `gcc`

Model inputs (committed under `models/deit_tiny_patch16_224/`, together with the FP32 export, the source checkpoint and the reference outputs):

```text
models/deit_tiny_patch16_224/deit_tiny_w8a8_pilot.tflite   SHA-256 d7c2cba3f8f5758e1680dda194e3471790b3b1bf64f83ce30ad7d9922089de26
models/deit_tiny_patch16_224/sample.png                    test image
```

The `.tflite` file is produced by `tools/model/export_deit_w8a8.py` from the pinned checkpoint
(`litert-community/deit_tiny_patch16_224`, revision `d0677ea57e9dfe6ed8d083d871562435ae90420a`); see
[`models/deit_tiny_patch16_224/README.md`](models/deit_tiny_patch16_224/README.md). Verify the hash before running anything:

```bash
sha256sum models/deit_tiny_patch16_224/deit_tiny_w8a8_pilot.tflite
```

One-time environment setup:

```bash
bash scripts/setup/setup_host_venv.sh          # ~/aca-venv: numpy 2.1.3, ai-edge-litert 2.2.0, matplotlib 3.11.2
bash scripts/setup/install_xqmac8_spike.sh     # builds the Spike plugin -> ~/riscv/libcustomext.so
```

## Running everything

Run from anywhere inside the repo (the scripts `cd` to the root). In order:

```bash
bash scripts/run/gen_check.sh 0:648     # 1. generate C for all 648 ops, compare every tensor to TFLite (host)
bash scripts/run/run_model.sh           # 2. generate, build and run B0 B1 X1 X1MAC on Spike; per-op counters
bash scripts/run/instr_mix.sh           # 3. whole-model instruction mix and code size (spike -g PC histogram)
bash scripts/run/verify_isa.sh          # 4. directed + randomized xqmac8 / xrequant tests
```

Expected output:

| Step | You should see |
|---|---|
| 1 | 0 mismatching tensors, in both no-reuse and reuse modes |
| 2 | `RESULT PASS` and `TOTAL instret=` B0 12598660515, B1 6478403747, X1 1522226359, X1MAC 1580552607 |
| 3 | `results/model/instr_mix.md` rewritten; totals within 0.3% of step 2 (the histogram also counts `main`) |
| 4 | three seeds, each `failures=0` and `RESULT PASS` |

Outputs: `results/model/{B0,B1,X1,X1MAC}.log`, `results/model/profile.md`, `results/model/instr_mix.md`,
`results/isa_verify.log`. Instruction counts are deterministic.

Useful variants:

```bash
bash scripts/run/run_model.sh 0:648 B0 X1                 # only some configurations
bash scripts/run/run_model.sh 8:60 X1                     # only an op range (here: block 0)
REF_VARIANT=flip   bash scripts/run/gen_check.sh 0:648    # mirrored test input (also: invert)
HOST_EXTRA=src/gen/ops_opt.c bash scripts/run/gen_check.sh 0:648   # check the B1 kernels on the host
ISA_VERIFY_N=2000000 bash scripts/run/verify_isa.sh 1 2 3 4         # more random cases / seeds
OPT=-O1 bash scripts/run/run_model.sh                     # different optimization level for all configs
```

Earlier, smaller experiments (single-operator tiles, the Block 0 MLP chain, synthetic GEMM) are recorded in
`EXPERIMENTS.md` and rerun with `scripts/run/run_deit_op.sh <op>`, `run_all_deit_ops.sh` and `run_deit_fc1_configs.sh`.
Note that those used B0 at `-O0`; the whole-model B0 is `-O2`, so the two sets of B0 numbers are not comparable.

## Tests and verification

| Test | Where | What it checks |
|---|---|---|
| `tests/test_xrequant_ref.c` | host, run by `verify_isa.sh` | `xrequant` oracle equals TFLite's fixed-point helpers over 20 M random and edge cases |
| `src/workloads/isa_verify_main.c` | on Spike, run by `verify_isa.sh` | every executed `xqmac8` / `xrequant` equals its C oracle: 864 + 59,094 directed and 2 x 200,000 random per seed |
| `tests/test_xqmac8.c` | host | `xqmac8` reference semantics |
| `tests/test_xqmac8_encoding.c`, `tests/test_xrequant_encoding.c` | host | instruction encodings |
| `tests/test_requantize.c`, `tests/test_int8_gemm.c` | host | scalar requantization and INT8 GEMM kernels |
| `tests/xqmac8_smoke.S` | Spike | smallest possible end-to-end use of the instruction (`scripts/build/build_xqmac8_smoke.sh`) |

Directed cases cover lane isolation, all lanes at -128 / 127, INT64 accumulator wraparound, `INT32_MIN x INT32_MIN`
saturation, shifts 0..31, exact rounding ties, and zero-point and INT8 saturation extremes. A deliberately broken
oracle is caught (1,575 failures), so the tests can fail. Last committed run: 459,958 checks per seed, seeds 1-3,
0 failures (`results/isa_verify.log`).

Run one host test by hand, for example:

```bash
gcc -std=c11 -Wall -Wextra -Werror -Isrc -Isrc/kernels src/kernels/xqmac8.c tests/test_xqmac8.c -o /tmp/test_xqmac8 && /tmp/test_xqmac8
```

Why both an oracle and a Spike plugin: the oracle (`src/kernels/xqmac8.c`, `xrequant_ref.h`) is plain C and defines
the intended behaviour; `src/spike_ext/xqmac8_spike.cc` is the simulated hardware. They are written separately so
one can catch bugs in the other. The same oracle can verify the RTL later. The final model output alone is not a
sufficient test: an early `xrequant` that rounded once instead of twice changed 940 of 1000 logits while the top-1
class stayed correct.

## Configurations

| ID | Meaning |
|---|---|
| B0 | RV64IMC, straightforward scalar kernels (`-O2`) |
| B1 | RV64IMC, software-optimized kernels: zero points folded out of the inner loop, 2x4 register blocking, BMM operand packing, im2col for CONV |
| X1 | B1 structure plus `xqmac8` dot products and `xrequant` requantization |
| X1MAC | X1 with `xqmac8` only (exact software requantization), to separate the two instructions |

B1 and X1 share source files, so the B1-to-X1 difference isolates the two instructions. Non-GEMM ops (LayerNorm
pieces, ADD/MUL, softmax, GELU, transposes) use the same reference kernels in every configuration.

## Limitations

- **Instruction counts only.** No cycles, CPI, cache or stall behaviour, Fmax or area yet; those need CV-Wally RTL and synthesis.
- **Model accuracy is not qualified.** The W8A8 model was calibrated on a single image. Our C matches the TFLite INT8 model
  bit-exactly; how well that INT8 model matches the FP32 model on a validation set is not measured.
- **B1 is a chosen baseline, not a proven optimum.** For example, packing two activations per `mul` is a possible further scalar step.
- **Known approximations in the converter:** one empirical RSQRT calibration constant, and rare off-by-one at BATCH_MATMUL and
  SOFTMAX rounding ties (none reach the logits). See `docs/GENERATOR.md`, "Findings".
- Scripts target WSL Ubuntu with the toolchain and Spike paths listed under Setup.

## Repository layout

| Path | Contents |
|---|---|
| `src/gen/` | Generated-model kernels (`ops_ref.c` B0, `ops_opt.c` + `gemm_core.h` B1/X1), `tflite_math.h`, target driver `target_main.c` |
| `src/kernels/` | ISA oracles, encodings and inline wrappers (`xqmac8*`, `xrequant*`), earlier hand-written INT8 kernels |
| `src/spike_ext/` | `xqmac8_spike.cc`: Spike extension implementing both instructions |
| `src/startup/`, `src/workloads/` | Startup code, linker scripts, earlier tile workloads, `isa_verify_main.c` |
| `tools/` | Host Python: `tflite2c` (converter), `model` (export, inspection), `extract`, `analysis` |
| `scripts/` | `setup`, `build`, `run` drivers |
| `tests/` | Host-built unit tests and the Spike smoke test |
| `models/` | Pinned model artifacts (W8A8 and FP32 `.tflite`, source checkpoint, sample image) and metadata |
| `results/` | Committed measurement logs and tables |
| `docs/` | Requirements, generator and layout docs, model summary, figures, presentation (`docs/presentation/`) |
| `DESIGN.md`, `EXPERIMENTS.md`, `AI_USAGE.md` | Design decisions, experiment records, AI-assistance statement |

More: [`docs/REPO_LAYOUT.md`](docs/REPO_LAYOUT.md), [`docs/GENERATOR.md`](docs/GENERATOR.md),
[`docs/MODEL_SUMMARY.md`](docs/MODEL_SUMMARY.md).
