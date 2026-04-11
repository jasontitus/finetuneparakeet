# WSL2 CUDA issues for NeMo TDT training (2026-04-09)

Documented during attempts to run `parakeet-tdt-0.6b-v3` fine-tuning
on a WSL2 Ubuntu 24.04 box with an RTX 3090. This is a reference for
anyone trying to train NeMo RNN-T/TDT models on WSL2.

## TL;DR

NeMo's TDT loss kernel uses **numba.cuda** to JIT-compile CUDA
kernels. numba's CUDA support has multiple WSL2-specific issues that
make training crash at the first step, even though inference and
`numba.cuda.detect()` work fine. The root causes are:

1. **numba uses ctypes to load `libcuda.so`** — on WSL2, the driver
   lives at `/usr/lib/wsl/lib/libcuda.so.1` (a proxy to the Windows
   GPU driver). If anything else installs a `libcuda.so` at a
   standard Linux path (e.g., `/usr/lib/x86_64-linux-gnu/`), numba
   loads the wrong one and gets 0 devices.
2. **numba's device list gets emptied** between initialization and
   first training step. NeMo's CUDA graph enable/disable cycle
   during sanity check appears to reset numba's internal state.
3. **nvJitLink / NVVM version mismatches** when pip-installed CUDA
   packages don't match each other or the driver version.

**What works on WSL2:** Swap NeMo's default numba-based TDT loss for
the pure PyTorch implementation (`tdt_pytorch`). This bypasses numba
entirely — no CUDA kernel JIT, no nvJitLink, no ctypes driver loading.
See `swap_to_pytorch_tdt_loss()` in `scripts/05_finetune.py`. The
PyTorch loss is ~3-4x slower per step but actually completes training.

**What also works:** Training on GCP with NVIDIA's DLVM image
(system-wide CUDA 12.8 matching PyTorch's cu128 build, consistent
everything). The numba TDT loss works natively there.

**What nearly worked on WSL2 (before the PyTorch loss fix):**
`pip install numba-cuda[cu12] cuda-python` with
`NUMBA_CUDA_USE_NVIDIA_BINDING=1` — this bypasses the ctypes driver
loading and fixes the device enumeration. It passed all unit tests
(including nohup + external_stream). But crashed during actual kernel
compilation with an nvJitLink version mismatch.

## Environment

- **OS:** WSL2, Ubuntu 24.04, kernel 6.6.87.2-microsoft-standard-WSL2
- **GPU:** NVIDIA GeForce RTX 3090 (CC 8.6, Ampere), 24 GB
- **Driver:** 595.97 (Windows-side), CUDA 13.2 capability
- **Python:** 3.11.15 (via uv)
- **PyTorch:** 2.8.0+cu128
- **NeMo:** 2.4.1
- **numba:** 0.60.0 (also tried 0.65.0)

## Issue 1: `libnvvm.so` not found

**Symptom:**
```
numba.cuda.cudadrv.error.NvvmSupportError: libNVVM cannot be found.
libnvvm.so: cannot open shared object file: No such file or directory
```

**Cause:** PyTorch bundles its own CUDA runtime but does NOT include
`libnvvm.so` or `libdevice.10.bc`. numba needs these for kernel
compilation.

**Fix:** Install the NVVM library via pip:
```bash
pip install nvidia-cuda-nvcc-cu12
```
Then set env vars:
```bash
NVCC_BASE=".venv/lib/python3.11/site-packages/nvidia/cuda_nvcc"
export CUDA_HOME="$NVCC_BASE"
export LD_LIBRARY_PATH="$NVCC_BASE/nvvm/lib64:$LD_LIBRARY_PATH"
export NUMBA_NVVM_LIBRARY="$NVCC_BASE/nvvm/lib64/libnvvm.so"
```

**Status:** Fixed. Numba detects GPU and NVVM version correctly.

## Issue 2: NVVM arch mismatch (`No supported GPU compute capabilities`)

**Symptom:**
```
NvvmSupportError: No supported GPU compute capabilities found.
Please check your cudatoolkit version matches your CUDA version.
```

**Cause:** numba 0.65.0's arch-selection table can't map the NVVM
version from `nvidia-cuda-nvcc-cu12==12.9.86` to compute capability
8.6 (RTX 3090).

**Fix:** Downgrade numba to 0.60.0 and nvcc to 12.4:
```bash
pip install "numba>=0.59,<0.61" "nvidia-cuda-nvcc-cu12>=12.4,<12.5"
```

**Status:** Fixed `detect()` and basic tests. But training still
crashed with Issue 3.

## Issue 3: Empty device list (`IndexError: list index out of range`)

**Symptom:**
```
File "numba/cuda/cudadrv/devices.py", line 40, in __getitem__
    return self.lst[devnum]
IndexError: list index out of range
```

This happens when NeMo's TDT loss calls
`cuda.external_stream(torch.cuda.current_stream(...).cuda_stream)`
during the first training step. Sanity check (inference only) passes.

**Cause:** numba uses ctypes to call `cuInit()` and
`cuDeviceGetCount()` from `libcuda.so`. On WSL2, the driver proxy at
`/usr/lib/wsl/lib/libcuda.so.1` works but numba's ctypes-based
loader sometimes finds a different (non-functional) `libcuda.so` or
the driver returns 0 devices through ctypes for reasons specific to
WSL2's IPC mechanism.

Reference: [numba/numba#9032](https://github.com/numba/numba/issues/9032),
[numba/numba#10191](https://github.com/numba/numba/issues/10191)

**Things that did NOT fix this:**
- `NUMBA_CUDA_DRIVER=/usr/lib/wsl/lib/libcuda.so.1`
- `CUDA_VISIBLE_DEVICES=0`
- `numba.cuda.select_device(0)` before training
- Monkey-patching `_DeviceList.__getitem__`
- Source-patching `numba/cuda/cudadrv/devices.py`
- Running under `tmux` instead of `nohup`

**What DID fix device enumeration:**
```bash
pip install numba-cuda[cu12] cuda-python
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
```

This makes numba use NVIDIA's `cuda-python` bindings instead of
ctypes. All tests pass including `external_stream()` under nohup.
BUT: actual kernel compilation then fails with Issue 4.

## Issue 4: nvJitLink version mismatch

**Symptom:**
```
cuda.bindings.nvjitlink.nvJitLinkError: ERROR_INTERNAL (6)
nvJitLink error log: ERROR 4 in nvvmAddNVVMContainerToProgram,
may need newer version of nvJitLink library
```

**Cause:** The pip packages installed by `numba-cuda[cu12]` pulled in
`nvidia-cuda-nvcc-cu12==12.9.86` which includes NVVM 12.9. The
nvJitLink library (also 12.9) fails during kernel compilation,
possibly due to a mismatch with the driver's CUDA version (13.2) or
an internal format incompatibility.

**Status:** Bypassed — see "Resolution" section below. The nvJitLink
issue itself remains unfixed, but switching to the pure PyTorch TDT
loss avoids numba kernel compilation entirely.

**Upstream tracking:**
- [numba/numba#10353](https://github.com/numba/numba/issues/10353) —
  same error on WSL2 (RTX 4090, numba 0.62.1). Closed but unclear if
  actually fixed.
- [NVIDIA-NeMo/NeMo#15155](https://github.com/NVIDIA-NeMo/NeMo/issues/15155) —
  same error with `numba-cuda>0.15.1`. Still open as of 2026-04-09.
  Workaround: `pip install numba-cuda==0.15.1`. NeMo PRs #15183 and
  #15166 merged to main but not yet released to PyPI.
- [numba/numba#9032](https://github.com/numba/numba/issues/9032) —
  broader WSL2 numba+CUDA incompatibility (ctypes driver loading).

**Recommended fix:** Pin `numba-cuda==0.15.1`:
```bash
uv pip install 'numba-cuda==0.15.1'
```
This keeps the fast numba TDT loss (~5-7 it/s on RTX 3090) vs the
pure PyTorch fallback (~0.01 it/s). The script auto-detects which
path works — see `maybe_swap_to_pytorch_tdt_loss()` in
`scripts/05_finetune.py`. Confirmed working 2026-04-10.

## Issue 5: `apt install nvidia-cuda-toolkit` breaks PyTorch

**Symptom:**
```
RuntimeError: No CUDA GPUs are available
```
after running `sudo apt install nvidia-cuda-toolkit`.

**Cause:** The apt package installs a Linux-native `libcuda.so` at
`/usr/lib/x86_64-linux-gnu/` which **shadows** the WSL2 Windows
driver proxy at `/usr/lib/wsl/lib/libcuda.so.1`. PyTorch links to
the wrong one and can't see the GPU.

**Fix:**
```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```
This makes the dynamic linker find the WSL2 driver FIRST. PyTorch
and numba both work again. BUT this doesn't fix the nvJitLink
compilation issue (Issue 4).

**Better fix:** Don't install `nvidia-cuda-toolkit` on WSL2 at all.
The apt package ships CUDA 12.0 which conflicts with everything else.
To undo:
```bash
sudo apt remove nvidia-cuda-toolkit
```

## Resolution: pure PyTorch TDT loss (2026-04-09)

NeMo ships two TDT loss implementations:

| Loss name      | Backend | Class              | Needs numba? |
|----------------|---------|--------------------|--------------|
| `tdt`          | numba   | `TDTLossNumba`     | Yes          |
| `tdt_pytorch`  | torch   | `TDTLossPytorch`   | No           |

The pretrained `parakeet-tdt-0.6b-v3` model defaults to `tdt` (numba).
Switching to `tdt_pytorch` after loading the model bypasses all of
Issues 1-4 because no numba CUDA kernels are ever compiled.

**What `scripts/05_finetune.py` does:**

`swap_to_pytorch_tdt_loss(model)` runs after `load_model()` and:

1. Reads the existing TDT kwargs (`durations`, `sigma`) from the
   model's config.
2. Creates a new `RNNTLoss` with `loss_name="tdt_pytorch"`.
3. Replaces **both** `model.loss` and `model.joint._loss`. The joint
   module caches a separate loss reference when `fuse_loss_wer` is
   enabled — if you only swap `model.loss`, `training_step` still
   dispatches through `joint.forward() → joint._loss` (the old numba
   one). This was the key gotcha.
4. Updates `model.cfg.loss` so checkpoint saving reflects the actual
   loss in use.

**Performance trade-off:** The PyTorch TDT loss is ~3-4x slower per
step than the numba version (~20-30s/step vs ~7s/step on an RTX 3090
with batch_size=2, encoder frozen). For the full 5-epoch run this
means ~6-8h wall time instead of ~2h. Acceptable for local WSL2
training; GCP with numba is still faster for large runs.

**Env vars needed on WSL2:**

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

No numba-related env vars are needed since numba is not used at all.

## What also works

### GCP DLVM image

NVIDIA's Deep Learning VM image ships with a consistent CUDA toolkit
(12.8) matching PyTorch's cu128 build. The numba TDT loss works
natively there (faster). Use `scripts/launch_ft_vm.sh` to spin up an
L4 on-demand VM.

### WSL2 for inference only

Baseline eval with `scripts/04_eval.py` works on WSL2 because NeMo's
inference path doesn't use numba CUDA kernels — it uses PyTorch's
decoder which links to CUDA differently.

### Docker on WSL2 (untested but likely works)

Running inside NVIDIA's NeMo container
(`nvcr.io/nvidia/nemo:24.xx`) with `--gpus all` should work because
the container ships its own complete, consistent CUDA toolkit.
