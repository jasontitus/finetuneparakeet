#!/bin/bash
# Eval-only pipeline for a fine-tuned model on all Lithuanian test sets.
# No training — just greedy + beam+LM evaluation.
#
# Usage:
#   bash scripts/run_eval_only.sh <model> [run_tag]
#
# Example:
#   bash scripts/run_eval_only.sh sliderforthewin/parakeet-tdt-lt ltft
set -e

MODEL="${1:?usage: $0 <model-id-or-path> [run_tag]}"
TAG="${2:-eval}"

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -d .venv ]; then
  source .venv/bin/activate
fi

echo "▶ model: $MODEL"
echo "▶ tag:   $TAG"
echo "▶ repo:  $REPO_ROOT"

# Build manifests if not already present
if [ ! -s data/manifests/cv25_lt_test.json ]; then
  echo "▶ building manifests"
  python scripts/03_prepare_manifests.py --datasets cv25_lt voxpopuli_lt fleurs_lt shunyalabs_lt
fi

# Datasets to evaluate (test splits only)
DATASETS="cv25_lt_test fleurs_lt_test voxpopuli_lt_test shunyalabs_lt_test"

# Greedy eval on each test set
for ds in $DATASETS; do
  manifest="data/manifests/${ds}.json"
  if [ ! -f "$manifest" ]; then
    echo "  SKIP $ds (no manifest)"
    continue
  fi
  out="results/${TAG}_${ds}_greedy"
  echo "▶ greedy eval: $ds"
  python scripts/04_eval.py \
    --model "$MODEL" \
    --manifest "$manifest" \
    --out "$out"
done

# Beam+LM eval on each test set (if LM available)
LM_FILE="data/lm/lt_token_4gram.arpa"
if [ -f "$LM_FILE" ]; then
  for ds in $DATASETS; do
    manifest="data/manifests/${ds}.json"
    if [ ! -f "$manifest" ]; then
      continue
    fi
    out="results/${TAG}_${ds}_beamlm"
    echo "▶ beam+LM eval: $ds (alpha=0.5)"
    python scripts/11_eval_beam_lm.py \
      --model "$MODEL" \
      --manifest "$manifest" \
      --lm "$LM_FILE" \
      --beam-size 8 --alpha 0.5 \
      --out "$out"
  done
else
  echo "▶ SKIP beam+LM (no LM at $LM_FILE)"
fi

echo "▶ DONE — results:"
for d in results/${TAG}_*; do
  if [ -f "$d/summary.json" ]; then
    echo "  $d:"
    python -c "import json; d=json.load(open('$d/summary.json')); print(f'    WER={d[\"wer\"]:.4f} CER={d[\"cer\"]:.4f} n={d[\"n_clips\"]}')"
  fi
done
