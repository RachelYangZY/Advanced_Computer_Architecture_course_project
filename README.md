# Quantized DeiT-Tiny on bare-metal RV64IMC with packed INT8 MAC and requantization instructions

Term project for Advanced Computer Architecture: accelerate W8A8 DeiT-Tiny inference on an RV64IMC core (CV-Wally)
with two custom instructions, `xqmac8` (8 x INT8 multiply-accumulate) and `xrequant` (fused requantization),
evaluated first on Spike (ISA level) and then in CV-Wally RTL.

## Status

| Stage | State |
|---|---|
| Model: DeiT-Tiny W8A8 (5.7 M params, 1.25 GMAC), converted to integer-only C by `tools/tflite2c` | Done, bit-exact vs TFLite (3 inputs) |
| B0, B1, X1, X1MAC kernel libraries, full inference on Spike | Done |
| `xqmac8`, `xrequant`: semantics, Spike implementation, directed + randomized verification | Done |
| ISA-level metrics: instruction mix, memory instructions, MAC/instr, code size, custom-instruction use | Done |
| CV-Wally RTL, synthesis (area, Fmax), extension | Planned |

Headline (Spike, one inference): B0 12.60 G instructions, B1 6.48 G (1.94x), **X1 1.52 G (8.28x vs B0, 4.26x vs B1)**,
0.82 MAC/instruction. Details: [`EXPERIMENTS.md`](EXPERIMENTS.md).

## Quick start (WSL Ubuntu)

```bash
bash scripts/setup/setup_host_venv.sh                 # Python env (numpy, ai-edge-litert, matplotlib)
bash scripts/setup/install_xqmac8_spike.sh            # build the Spike extension (needs ~/riscv-isa-sim)
bash scripts/run/gen_check.sh 0:648                   # generate C, compare every tensor to TFLite on the host
bash scripts/run/run_model.sh                         # build + run B0/B1/X1/X1MAC on Spike, per-op counters
bash scripts/run/instr_mix.sh                         # whole-model instruction mix (PC histogram), code size
bash scripts/run/verify_isa.sh                        # directed + randomized xqmac8/xrequant tests
```

## Layout

| Path | Contents |
|---|---|
| `src/gen/` | Generated-model kernels (`ops_ref.c` B0, `ops_opt.c` + `gemm_core.h` B1/X1), target driver |
| `src/kernels/`, `src/spike_ext/` | ISA oracles and encodings, Spike extension |
| `src/startup/`, `src/workloads/` | Startup code, linker scripts, earlier tile workloads, ISA verification program |
| `tools/` | Host-side Python: `tflite2c` (generator), `analysis`, `extract`, `model` |
| `scripts/` | `setup`, `build`, `run` drivers |
| `docs/` | Requirements, generator and layout docs, model summary |
| `results/` | Committed measurement logs and tables |
| `DESIGN.md`, `EXPERIMENTS.md`, `AI_USAGE.md` | Design decisions, experiment records, AI-assistance statement |

See [`docs/REPO_LAYOUT.md`](docs/REPO_LAYOUT.md) and [`docs/GENERATOR.md`](docs/GENERATOR.md).
