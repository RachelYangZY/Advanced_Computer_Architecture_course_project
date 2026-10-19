# Repository Layout

Rule of thumb: **`src/` runs on the RISC-V target (or inside Spike), `tools/` are host Python programs
that produce or analyze things, `scripts/` are shell drivers that build and run.**

| Directory | Purpose | Runs where |
|---|---|---|
| `src/` | C/asm that ends up in a bare-metal ELF, plus the Spike extension. | Target / Spike |
| `tools/` | Python (and one awk) programs: parse the model, generate C, extract data, analyze traces. | Host (WSL venv) |
| `scripts/` | Bash drivers: set up, cross-compile B0/B1/X1, run Spike, run checks. | Host (WSL) |
| `tests/` | Host-built unit tests for kernels and instruction encodings. | Host |
| `docs/`, `*.md` | Requirements, design, experiments, model/generator docs. | - |
| `models/`, `results/` | Model files and metadata; measured metrics. | - |

## src/

| Subdir | Contents |
|---|---|
| `src/kernels/` | Hand-written INT8 kernels and custom-instruction wrappers: `int8_gemm*`, `requantize`, `xqmac8*`, `xrequant*`. |
| `src/gen/` | Reference kernels used by the generated model (`ops_ref`, `tflite_math.h`). |
| `src/workloads/` | `main()` programs and generated data headers for the hand-extracted DeiT tiles/chain (`deit_*`, `b0_main.c`, `x1_main.c`). |
| `src/startup/` | `start.S` and the linker script `b0.ld`, shared by all configurations. |
| `src/spike_ext/` | `xqmac8_spike.cc` (Spike extension: xqmac8 and xrequant). |

## tools/

| Subdir | Contents |
|---|---|
| `tools/tflite2c/` | The converter: `tflite_ir.py` (parse), `gen_model.py` (lower, plan, emit), `dump_ref_tensors.py` (TFLite reference), `summarize_model.py`. |
| `tools/extract/` | Per-operator extractors that predate the generator (`extract_deit_*`, `generate_fc1_requant_params.py`). |
| `tools/model/` | Model export and inspection (`export_deit_w8a8.py`, `inspect_*`, `probe_*`, `list_w8a8_weights.py`, `run_deit_reference.py`). |
| `tools/analysis/` | Spike trace and metrics analysis (`spike_metrics.py`, `summarize_deit_fc1_metrics.py`, `analyze_kernel_trace.awk`). |

## scripts/

| Subdir | Contents |
|---|---|
| `scripts/setup/` | One-time environment setup: `setup_host_venv.sh`, `install_xqmac8_spike.sh`. |
| `scripts/build/` | Cross-compile one configuration: `build_{b0,b1,x1}*.sh`, `build_deit_op.sh <op> <cfg>`. |
| `scripts/run/` | Run flows: `run_deit_op.sh <op>`, `run_all_deit_ops.sh`, `gen_check.sh <ops>`, `run_xrequant_fc1.sh`, result checks. |

All scripts are run from anywhere (they `cd` to the repo root) and use the repo-root-relative paths above.
Include paths for target builds are `-Isrc/kernels -Isrc/workloads`; the generator test uses `-Isrc/gen`.
