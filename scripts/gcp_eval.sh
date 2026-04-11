#!/usr/bin/env bash
# Self-contained GCP VM eval runner for parakeet-tdt-lt.
#
# Assumes a GCP Deep Learning VM with PyTorch + CUDA already installed
# (like the `pytorch-2-8-cu128-ubuntu-2204-nvidia-570` image). On a fresh
# VM:
#
#     gcloud compute ssh <vm> -- bash
#     # then on the VM:
#     git clone https://github.com/jasontitus/finetuneparakeet.git
#     cd finetuneparakeet
#     bash scripts/gcp_eval.sh
#
# Writes results to ./results/gcp_eval_<timestamp>/
set -euo pipefail

OUT_BASE="${OUT_BASE:-results/gcp_eval_$(date +%Y%m%d-%H%M%S)}"
MODEL="${MODEL:-sliderforthewin/parakeet-tdt-lt}"
BATCH_SIZE="${BATCH_SIZE:-16}"

echo "▸ OUT_BASE: $OUT_BASE"
echo "▸ MODEL:    $MODEL"
mkdir -p "$OUT_BASE"

# ── 1. Install deps ──────────────────────────────────────────────────
echo ""
echo "▸ installing Python deps (this takes a few minutes)..."
pip install --quiet --upgrade pip
pip install --quiet \
    'nemo_toolkit[asr]>=2.0,<2.5' \
    'numba-cuda==0.15.1' \
    'datasets>=2.18,<4' \
    'jiwer>=3.0' \
    'huggingface_hub' \
    'https://github.com/kpu/kenlm/archive/master.zip'

# ── 2. Get CV25 LT test data ─────────────────────────────────────────
# The manifests in this repo use absolute paths from the machine where
# they were built. We need to either (a) rebuild them on this VM, or
# (b) patch the paths. Option (a) is cleaner.
echo ""
echo "▸ preparing CV25 LT test data..."
if [ ! -d "data/cv25_lt/cv-corpus-25.0-2026-03-09" ]; then
    mkdir -p data/cv25_lt
    # Pull from the GCS bucket the project uses
    if command -v gsutil >/dev/null 2>&1; then
        echo "  downloading CV25 LT tarball from GCS..."
        gsutil -q cp \
            gs://safecare-maps-speechbench/corpora/cv25-lt/cv-corpus-25.0-2026-03-09-lt.tar.gz \
            data/cv25_lt/cv25_lt.tar.gz
        echo "  extracting..."
        tar -xzf data/cv25_lt/cv25_lt.tar.gz -C data/cv25_lt
        rm -f data/cv25_lt/cv25_lt.tar.gz
    else
        echo "  gsutil not found; cannot fetch CV25 tarball automatically"
        echo "  please place the extracted corpus under data/cv25_lt/cv-corpus-25.0-2026-03-09/"
        exit 1
    fi
fi

# ── 3. Rebuild manifests on this VM (so absolute paths match) ────────
echo ""
echo "▸ rebuilding manifests with this VM's absolute paths..."
python scripts/03_prepare_manifests.py \
    --out data/manifests \
    --datasets cv25_lt fleurs_lt voxpopuli_lt

# ── 4. Download the LM from HF ───────────────────────────────────────
echo ""
echo "▸ fetching LM from HuggingFace..."
mkdir -p data/lm
python -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id='$MODEL',
    filename='lt_token_4gram.arpa',
    local_dir='data/lm',
)
print(p)
"

# ── 5. Greedy eval on CV25 LT test ───────────────────────────────────
echo ""
echo "▸ === greedy eval: CV25 LT test ==="
python scripts/04_eval.py \
    --model "$MODEL" \
    --manifest data/manifests/cv25_lt_test.json \
    --out "$OUT_BASE/cv25_test_greedy" \
    --batch-size "$BATCH_SIZE"

# ── 6. Greedy eval on FLEURS LT test ─────────────────────────────────
echo ""
echo "▸ === greedy eval: FLEURS LT test ==="
python scripts/04_eval.py \
    --model "$MODEL" \
    --manifest data/manifests/fleurs_lt_test.json \
    --out "$OUT_BASE/fleurs_test_greedy" \
    --batch-size "$BATCH_SIZE"

# ── 7. Beam + LM eval on CV25 LT test (alpha=0.5 best for CV25) ──────
echo ""
echo "▸ === beam + LM eval: CV25 LT test (alpha=0.5) ==="
python scripts/11_eval_beam_lm.py \
    --model "$MODEL" \
    --manifest data/manifests/cv25_lt_test.json \
    --lm data/lm/lt_token_4gram.arpa \
    --beam-size 4 \
    --alpha 0.5 \
    --out "$OUT_BASE/cv25_test_beamlm_a05" \
    --batch-size "$BATCH_SIZE"

# ── 8. Beam + LM eval on FLEURS LT test (alpha=0.3 best for FLEURS) ──
echo ""
echo "▸ === beam + LM eval: FLEURS LT test (alpha=0.3) ==="
python scripts/11_eval_beam_lm.py \
    --model "$MODEL" \
    --manifest data/manifests/fleurs_lt_test.json \
    --lm data/lm/lt_token_4gram.arpa \
    --beam-size 4 \
    --alpha 0.3 \
    --out "$OUT_BASE/fleurs_test_beamlm_a03" \
    --batch-size "$BATCH_SIZE"

# ── 9. Summary ───────────────────────────────────────────────────────
echo ""
echo "▸ === summary ==="
for dir in "$OUT_BASE"/*/; do
    name=$(basename "$dir")
    if [ -f "$dir/summary.json" ]; then
        wer=$(python -c "import json; d=json.load(open('$dir/summary.json')); print(f\"{d['WER']*100:.2f}\" if d['WER']<1 else f\"{d['WER']:.2f}\")" 2>/dev/null || echo "?")
        cer=$(python -c "import json; d=json.load(open('$dir/summary.json')); print(f\"{d['CER']*100:.2f}\" if d['CER']<1 else f\"{d['CER']:.2f}\")" 2>/dev/null || echo "?")
        n=$(python -c "import json; d=json.load(open('$dir/summary.json')); print(d['n_clips'])" 2>/dev/null || echo "?")
        printf "  %-35s  n=%-6s  WER=%s%%  CER=%s%%\n" "$name" "$n" "$wer" "$cer"
    fi
done

echo ""
echo "▸ done. results in $OUT_BASE/"
