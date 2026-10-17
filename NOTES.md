# DeiT-Tiny Bare-Metal Port Notes

## 2026-09-29: Environment/API Gate

### Host and target tools

- Active host shell: Ubuntu 24.04 WSL, Python 3.12.3.
- RISC-V compiler target: `riscv64-unknown-elf`.
- Spike: `Spike RISC-V ISA Simulator 1.1.1-dev` at
  `/home/rachelyzy/riscv/bin/spike`.
- Spike default ISA from `spike --help`: `rv64imafdc_zicntr_zihpm`.
- Existing project bare-metal startup: `src/start.S` initializes `sp` and calls
  `main`.
- Existing linker script: `scripts/b0.ld`, entry `_start`, load base
  `0x80000000`, 4 KiB stack allocation, `.tohost` section.
- Existing completion convention: write `1` for pass or `3` for failure to
  `tohost`, then remain in a loop.
- Do not assume a TFLM firmware runtime, Wally, or its linker/startup is present.

### Verified LiteRT names and versions

- PyPI stable versions found on 2026-09-29:
  - `litert-torch==0.9.4`; import as `litert_torch`.
  - `ai-edge-quantizer==0.9.0`; imports from `ai_edge_quantizer`.
  - `ai-edge-litert==2.2.0`; LiteRT host interpreter/runtime package.
- Current LiteRT Torch API per official documentation:
  `litert_torch.convert(model.eval(), sample_inputs)` followed by
  `edge_model.export(path)`.
- LiteRT Torch documents Linux support, Python >=3.10 (Python 3.11 recommended),
  with PyTorch and `tf-nightly` dependencies.
- AI Edge Quantizer documents Linux/macOS and Python 3.10-3.13. The static
  `recipe.static_wi8_ai8()` recipe is the documented W8A8 recipe; calibration is
  required. Stable PyPI is 0.9.0; docs also refer to `ai-edge-litert-nightly`.
- Package inventory/API checks used:
  `python3 -m pip index versions litert-torch`
  `python3 -m pip index versions ai-edge-quantizer`
  `python3 -m pip index versions ai-edge-litert`
  Official references: LiteRT Torch README and AI Edge Quantizer README/operator
  coverage at `github.com/google-ai-edge/litert-torch` and
  `github.com/google-ai-edge/ai-edge-quantizer`.

### Environment creation check

- Attempted:
  `python3 -m venv models/deit_tiny_patch16_224/.venv`
- Result: **failed**. Ubuntu reports `ensurepip is not available` and requests
  `python3.12-venv`; the partially created directory has no pip. No model
  packages have been installed into the system Python.
- Decision: bootstrap `virtualenv` into `/tmp` without sudo, create a local
  project venv, and rerun the venv/pip check. Do not install requirements until
  that check passes.

### Model and conversion status

- DeiT-Tiny LiteRT artifact at the pinned Hugging Face revision is FLOAT32, not
  W8A8; its structural audit is in
  `models/deit_tiny_patch16_224/README.md`.
- No `.tflite` model-preparation stages have been treated as complete yet.
- Keep the requested fixed 1,000-image ImageNet validation subset separate from
  the 200-500-image calibration subset. Record dataset source/access and exact
  sample IDs before running accuracy comparisons.

## 2026-10-14: Host venv and first DeiT GEMM tile

- Native Python packages must not be installed under the OneDrive/WSL mount;
  use `/home/rachelyzy/aca-venv` (`scripts/setup_host_venv.sh`).
- Extract/build/run path for the first model-shaped tile:
  `scripts/extract_deit_gemm_tile.py`
  `scripts/build_{b0,b1,x1}_deit_fc1.sh`
  `scripts/run_deit_fc1_configs.sh`
  `scripts/summarize_deit_fc1_metrics.py`
- Spike X1 command requires `LD_LIBRARY_PATH=$HOME/riscv:$HOME/riscv/lib` and
  `--extension=xqmac8`.
- Compatibility note: the original synthetic B0/B1/X1 GEMM path remains valid.
  X1 was generalized with an `inner` argument so the same kernel serves both
  `synthetic-v1` (K=16) and `deit-block0-mlp-fc1-tile-v1` (K=192).