#!/usr/bin/env bash
# Run Open ASR Leaderboard-compatible eval on Lithuanian datasets.
#
# Uses whisper BasicTextNormalizer (the leaderboard standard) and loads
# datasets directly from HuggingFace to ensure identical test splits.
#
# Run on a GCP VM with GPU:
#     pip install whisper-normalizer
#     bash scripts/gcp_leaderboard_eval.sh
#
# Evaluates BOTH the fine-tuned model AND the baseline for comparison.
set -euo pipefail

MODEL="${MODEL:-sliderforthewin/parakeet-tdt-lt}"
BASELINE="${BASELINE:-nvidia/parakeet-tdt-0.6b-v3}"
OUT="${OUT:-results/leaderboard_$(date +%Y%m%d)}"

echo "▸ model:    $MODEL"
echo "▸ baseline: $BASELINE"
echo "▸ output:   $OUT"

pip install --quiet whisper-normalizer 2>/dev/null || true

# ── Fine-tuned model ─────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════"
echo "  Fine-tuned: $MODEL"
echo "════════════════════════════════════════"

echo "▸ FLEURS lt_lt"
python scripts/12_eval_leaderboard.py \
    --model "$MODEL" \
    --dataset google/fleurs --config lt_lt --split test \
    --text-field transcription \
    --out "$OUT/ft_fleurs_lt"

echo "▸ VoxPopuli lt"
python scripts/12_eval_leaderboard.py \
    --model "$MODEL" \
    --dataset facebook/voxpopuli --config lt --split test \
    --text-field normalized_text \
    --out "$OUT/ft_voxpopuli_lt"

# ── Baseline model ───────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════"
echo "  Baseline: $BASELINE"
echo "════════════════════════════════════════"

echo "▸ FLEURS lt_lt"
python scripts/12_eval_leaderboard.py \
    --model "$BASELINE" \
    --dataset google/fleurs --config lt_lt --split test \
    --text-field transcription \
    --out "$OUT/baseline_fleurs_lt"

echo "▸ VoxPopuli lt"
python scripts/12_eval_leaderboard.py \
    --model "$BASELINE" \
    --dataset facebook/voxpopuli --config lt --split test \
    --text-field normalized_text \
    --out "$OUT/baseline_voxpopuli_lt"

# ── Summary ──────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════"
echo "  Summary (BasicTextNormalizer)"
echo "════════════════════════════════════════"
printf "%-35s  %8s  %8s  %6s\n" "eval" "WER" "CER" "n"
echo "────────────────────────────────────────────────────────────────"
for d in "$OUT"/*/; do
    name=$(basename "$d")
    if [ -f "$d/summary.json" ]; then
        python3 -c "
import json
d = json.load(open('$d/summary.json'))
wer = d['wer']*100 if d['wer'] < 1 else d['wer']
cer = d['cer']*100 if d['cer'] < 1 else d['cer']
print(f'  {\"$name\":<33}  {wer:>7.2f}%  {cer:>7.2f}%  {d[\"n_clips\"]:>5}')
" 2>/dev/null
    fi
done

echo ""
echo "▸ done. Upload model card to HF and submit at:"
echo "  https://huggingface.co/spaces/hf-audio/open_asr_leaderboard"
echo ""
echo "  The model-index in hf_model_card/README.md has the metadata"
echo "  the leaderboard auto-evaluator needs. Copy it to your HF"
echo "  model's README.md, update the WER numbers with the"
echo "  BasicTextNormalizer results above, and submit."
