# AI-Assisted Work

An LLM coding assistant (Cursor) was used throughout this project. The project owner is responsible for verifying
all technical decisions, measurements and generated code; every reported number comes from the committed scripts.

## What the assistant did

- **Model export and pilot quantization.** Repaired the DeiT-Tiny LiteRT/AI Edge Quantizer export flow, verified FP32 parity, and
  audited a one-image W8A8 pilot. The pilot's single-sample accuracy is not a representative validation result;
  calibration and evaluation on separate authorized data remain to be done.
- **TFLite-to-C generator** (`tools/tflite2c`, `src/gen/ops_ref.*`): parser, memory planner, operator lowering, host
  verification against the TFLite interpreter, and the investigation of TFLite's rounding behavior (including the three
  empirical choices listed in `docs/GENERATOR.md`).
- **Kernels and ISA**: B1/X1 kernels, the `xrequant` TFLite-exact modes, directed and randomized ISA verification
  (`src/workloads/isa_verify_main.c`).
- **Analysis and documents**: per-op profiling, PC-histogram instruction-mix tooling, model-to-kernel visualization,
  and the repository documentation.

## What the team must verify

- The instruction semantics and encodings in `DESIGN.md` are the team's design decisions; the assistant proposed and implemented them.
- Reported numbers: re-run `scripts/run/run_model.sh`, `instr_mix.sh` and `verify_isa.sh` and confirm the numbers.
- The empirical generator choices (RSQRT bias, MEAN rounding rule, integer softmax) reproduce TFLite on the tested inputs only.