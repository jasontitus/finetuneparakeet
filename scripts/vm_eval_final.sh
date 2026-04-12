#!/bin/bash
# FINAL eval-only VM startup script. All models from GCS, no HuggingFace.
# Includes: greedy + beam+LM on CV25/FLEURS/VoxPopuli, both normalizers.
set -euo pipefail

LOG=/var/log/ft-eval.log
exec > >(tee -a "$LOG") 2>&1
BUCKET="gs://safecare-maps-speechbench"
RUN_ID="$(curl -fsS -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/attributes/run-id 2>/dev/null || echo eval)"
VM="$(hostname)"
REMOTE_LOG="$BUCKET/finetune/$RUN_ID/logs/${VM}.eval.log"

# Log uploader — runs from SECOND 1, uploads every 60s
(while true; do sleep 60; gsutil -q cp "$LOG" "$REMOTE_LOG" 2>/dev/null || true; done) &
UPLOADER=$!
echo "▶ log uploader pid=$UPLOADER (every 60s → $REMOTE_LOG)"

on_exit() {
  kill $UPLOADER 2>/dev/null || true
  gsutil -q cp "$LOG" "$REMOTE_LOG" 2>/dev/null || true
  if [ "${1:-0}" -ne 0 ]; then
    echo "failed" | gsutil -q cp - "$BUCKET/finetune/$RUN_ID/logs/${VM}.failed" 2>/dev/null || true
  else
    echo "done" | gsutil -q cp - "$BUCKET/finetune/$RUN_ID/logs/${VM}.done" 2>/dev/null || true
  fi
}
trap 'on_exit 1' ERR
trap 'on_exit 0' EXIT

echo "▶ $(date -u) — eval-only startup"

# Deps
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg libsndfile1 git >/dev/null 2>&1 || true

echo "▶ cloning repo"
cd /opt
git clone --depth 1 https://github.com/jasontitus/finetuneparakeet.git
cd /opt/finetuneparakeet
ln -sf "$(which python3)" /usr/local/bin/python
sed -i "s/'numba-cuda==0.15.1'//" scripts/gcp_eval.sh || true

echo "▶ installing deps"
pip install --quiet --upgrade pip
pip install --quiet \
  'nemo_toolkit[asr]>=2.0,<2.5' \
  'datasets>=2.18,<4' \
  'jiwer>=3.0' \
  'huggingface_hub' \
  'whisper-normalizer'
echo "  deps done"

# Pull EVERYTHING from GCS — no HuggingFace model downloads
echo "▶ pulling models + LM from GCS (not HuggingFace)"
mkdir -p data/lm
gsutil -q cp "$BUCKET/corpora/models/parakeet-tdt-lt.nemo" ./ft.nemo
echo "  ft.nemo: $(du -sh ft.nemo | cut -f1)"
gsutil -q cp "$BUCKET/corpora/models/parakeet-tdt-0.6b-v3.nemo" ./baseline.nemo
echo "  baseline.nemo: $(du -sh baseline.nemo | cut -f1)"
gsutil -q cp "$BUCKET/corpora/lm/lt_token_4gram.arpa" data/lm/lt_token_4gram.arpa
echo "  LM: $(du -sh data/lm/lt_token_4gram.arpa | cut -f1)"
gsutil -q cp "$BUCKET/corpora/scripts/12_eval_leaderboard.py" scripts/12_eval_leaderboard.py
echo "  all cached ✓"

# Pull CV25 + build manifests
echo "▶ preparing data"
mkdir -p data/cv25_lt
gsutil -q cp "$BUCKET/corpora/cv25-lt/cv-corpus-25.0-2026-03-09-lt.tar.gz" data/cv25_lt/cv25.tar.gz
tar -xzf data/cv25_lt/cv25.tar.gz -C data/cv25_lt
rm data/cv25_lt/cv25.tar.gz
python scripts/03_prepare_manifests.py --datasets cv25_lt voxpopuli_lt fleurs_lt
echo "  manifests done ✓"

OUT="results/gcp_${RUN_ID}"
LM="data/lm/lt_token_4gram.arpa"

echo "▶ verifying CUDA"
python -c "import torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# ═══════════════════════════════════════════════════════════════
# EVALUATIONS — greedy + beam+LM, both models, all LT datasets
# ═══════════════════════════════════════════════════════════════

for MODEL_TAG in "ft:./ft.nemo" "baseline:./baseline.nemo"; do
  TAG="${MODEL_TAG%%:*}"
  MODEL="${MODEL_TAG##*:}"
  echo ""
  echo "════════════════════════════════════════"
  echo "  $TAG ($MODEL)"
  echo "════════════════════════════════════════"

  # Greedy eval on each test set
  for DS in cv25_lt_test fleurs_lt_test voxpopuli_lt_test; do
    MANIFEST="data/manifests/${DS}.json"
    [ -f "$MANIFEST" ] || continue
    echo "▶ greedy: $TAG / $DS"
    python scripts/04_eval.py --model "$MODEL" --manifest "$MANIFEST" --out "$OUT/${TAG}_${DS}_greedy" --batch-size 16
  done

  # Beam+LM on each test set (alpha=0.5 — user's best config)
  for DS in cv25_lt_test fleurs_lt_test voxpopuli_lt_test; do
    MANIFEST="data/manifests/${DS}.json"
    [ -f "$MANIFEST" ] || continue
    echo "▶ beam+LM α=0.5: $TAG / $DS"
    python scripts/11_eval_beam_lm.py --model "$MODEL" --manifest "$MANIFEST" --lm "$LM" --beam-size 4 --alpha 0.5 --out "$OUT/${TAG}_${DS}_beamlm"
  done
done

# ═══════════════════════════════════════════════════════════════
# LEADERBOARD-COMPATIBLE (BasicTextNormalizer) — best config only
# ═══════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════"
echo "  Leaderboard (BasicTextNormalizer)"
echo "════════════════════════════════════════"

for MODEL_TAG in "ft:./ft.nemo" "baseline:./baseline.nemo"; do
  TAG="${MODEL_TAG%%:*}"
  MODEL="${MODEL_TAG##*:}"

  # Greedy + beam+LM on FLEURS and VoxPopuli (loaded from HF datasets directly)
  for DS_CFG in "google/fleurs:lt_lt:transcription:fleurs_lt" "facebook/voxpopuli:lt:normalized_text:voxpopuli_lt"; do
    IFS=: read -r DS CFG TF NAME <<< "$DS_CFG"

    echo "▶ leaderboard greedy: $TAG / $NAME"
    python scripts/12_eval_leaderboard.py --model "$MODEL" --dataset "$DS" --config "$CFG" --split test --text-field "$TF" --out "$OUT/lb_${TAG}_${NAME}_greedy"

    echo "▶ leaderboard beam+LM α=0.5: $TAG / $NAME"
    python scripts/12_eval_leaderboard.py --model "$MODEL" --dataset "$DS" --config "$CFG" --split test --text-field "$TF" --lm "$LM" --beam-size 4 --alpha 0.5 --out "$OUT/lb_${TAG}_${NAME}_beamlm"
  done
done

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════"
echo "  SUMMARY"
echo "════════════════════════════════════════"
for d in "$OUT"/*/; do
  name=$(basename "$d")
  [ -f "$d/summary.json" ] || continue
  python3 -c "
import json; d=json.load(open('$d/summary.json'))
wer=d.get('wer',d.get('WER',0)); cer=d.get('cer',d.get('CER',0))
wer=wer*100 if wer<1 else wer; cer=cer*100 if cer<1 else cer
n=d.get('n_clips',0)
print(f'  {\"$name\":<40} WER={wer:>6.2f}%  CER={cer:>6.2f}%  n={n}')
" 2>/dev/null || true
done

echo ""
echo "▶ uploading results"
gsutil -q -m cp -r "$OUT" "$BUCKET/finetune/$RUN_ID/results/" || true
echo "▶ DONE at $(date -u)"
