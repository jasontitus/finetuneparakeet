#!/bin/bash
# Host-side pipeline driver for running the parakeet-LT fine-tune on a
# local (or SSH-accessible) Linux box with CUDA. Replaces the GCP VM
# startup script with a simpler flow that assumes the project is
# already on disk and the venv is already set up.
#
# Usage:
#   bash scripts/run_pipeline.sh [smoke|full]
#
# Modes match scripts/vm_startup.sh:
#   smoke - CV25 LT only, 1 epoch cap, small eval sample.
#   full  - all 4 LT datasets, 5 epochs, full test eval.
#
# Designed to be run under nohup so SSH disconnection doesn't kill it:
#   nohup bash scripts/run_pipeline.sh full > run.log 2>&1 &
#
# Output lands in ./results/ and ./checkpoints/, and a run-level log
# gets appended to ./run.log (or wherever you redirect stdout).
set -e

MODE="${1:-full}"
if [ "$MODE" != "smoke" ] && [ "$MODE" != "full" ]; then
  echo "usage: $0 smoke|full" >&2
  exit 2
fi

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
export PATH="$HOME/.local/bin:$PATH"

# Reduce CUDA fragmentation-related OOMs (same as VM startup).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -d .venv ]; then
  echo "✗ .venv missing — run the env setup first" >&2
  exit 3
fi
source .venv/bin/activate

# Numba needs libnvvm + libdevice, which PyTorch's CUDA bundle does NOT
# include. We pip-install nvidia-cuda-nvcc-cu12 (see requirements
# install block) which drops them under .venv/.../nvidia/cuda_nvcc/,
# and then point numba at them via the legacy NUMBAPRO_* env vars plus
# CUDA_HOME. Without this, NeMo's RNN-T (and TDT) loss fails with
# `libnvvm.so: cannot open shared object file`.
NVCC_BASE="$REPO_ROOT/.venv/lib/python3.11/site-packages/nvidia/cuda_nvcc"
if [ -f "$NVCC_BASE/nvvm/lib64/libnvvm.so" ]; then
  export CUDA_HOME="$NVCC_BASE"
  export NUMBAPRO_NVVM="$NVCC_BASE/nvvm/lib64/libnvvm.so"
  export NUMBAPRO_LIBDEVICE="$NVCC_BASE/nvvm/libdevice"
  export PATH="$NVCC_BASE/bin:$PATH"
fi

echo "▶ host:   $(hostname)"
echo "▶ mode:   $MODE"
echo "▶ python: $(python --version)"
echo "▶ cuda:   $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")')"
echo "▶ libnvvm: ${NUMBAPRO_NVVM:-UNSET}"
echo "▶ repo:   $REPO_ROOT"
echo

case "$MODE" in
  smoke)
    PREP_ARGS="--datasets cv25_lt"
    TRAIN_ARGS="--epochs 1"
    TRAIN_MANIFEST="data/manifests/cv25_lt_train.json"
    VAL_MANIFEST="data/manifests/cv25_lt_dev.json"
    TEST_MANIFEST="data/manifests/cv25_lt_test.json"
    BASELINE_EVAL_CAP="--max-clips 500"
    POST_EVAL_CAP="--max-clips 500"
    ;;
  full)
    PREP_ARGS="--datasets cv25_lt voxpopuli_lt fleurs_lt shunyalabs_lt"
    TRAIN_ARGS="--epochs 5"
    TRAIN_MANIFEST="data/manifests/ALL_train.json"
    VAL_MANIFEST="data/manifests/cv25_lt_dev.json"
    TEST_MANIFEST="data/manifests/cv25_lt_test.json"
    BASELINE_EVAL_CAP=""
    POST_EVAL_CAP=""
    ;;
esac

if [ -s data/manifests/ALL_train.json ]; then
  echo "▶ Step 1: build manifests — SKIP (ALL_train.json exists)"
else
  echo "▶ Step 1: build manifests"
  python scripts/03_prepare_manifests.py $PREP_ARGS
fi
echo

if [ -s results/baseline_cv25_lt_test/summary.json ]; then
  echo "▶ Step 2: baseline eval — SKIP (summary.json exists)"
  cat results/baseline_cv25_lt_test/summary.json | python -c "import json,sys;d=json.load(sys.stdin);print(f'  cached baseline: WER={d[\"wer\"]:.4f} CER={d[\"cer\"]:.4f} n={d[\"n_clips\"]}')"
else
  echo "▶ Step 2: baseline eval (pretrained parakeet-tdt-0.6b-v3)"
  python scripts/04_eval.py \
    --model nvidia/parakeet-tdt-0.6b-v3 \
    --manifest "$TEST_MANIFEST" \
    --out results/baseline_cv25_lt_test \
    $BASELINE_EVAL_CAP
fi
echo

echo "▶ Step 3: fine-tune"
python scripts/05_finetune.py \
  --config configs/finetune_lt.yaml \
  --model nvidia/parakeet-tdt-0.6b-v3 \
  --out-dir checkpoints/lt-ft \
  --train-manifest "$TRAIN_MANIFEST" \
  --val-manifest "$VAL_MANIFEST" \
  $TRAIN_ARGS
echo

echo "▶ Step 4: post-training eval"
python scripts/04_eval.py \
  --model checkpoints/lt-ft/finetuned.nemo \
  --manifest "$TEST_MANIFEST" \
  --out results/finetuned_cv25_lt_test \
  $POST_EVAL_CAP
echo

echo "▶ DONE"
echo "  baseline: $(cat results/baseline_cv25_lt_test/summary.json 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f"WER={d[\"wer\"]:.4f} CER={d[\"cer\"]:.4f}")' 2>/dev/null || echo '?')"
echo "  finetuned: $(cat results/finetuned_cv25_lt_test/summary.json 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f"WER={d[\"wer\"]:.4f} CER={d[\"cer\"]:.4f}")' 2>/dev/null || echo '?')"
