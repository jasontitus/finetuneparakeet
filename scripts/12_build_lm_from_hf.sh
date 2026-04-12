#!/usr/bin/env bash
# Build a token-level 4-gram LM for parakeet from the lt-asr-lm HF corpora.
#
# Downloads the combined Lithuanian text corpus from HuggingFace
# (sliderforthewin/lt-asr-lm-corpora), decompresses and concatenates all
# sources, then runs 08b_build_token_lm.py to build a token-level ARPA
# file compatible with NeMo's maes beam decoder.
#
# Prerequisites:
#   pip install huggingface_hub zstandard  (for download + decompress)
#   NeMo must be installed (for the tokenizer in 08b_build_token_lm.py)
#
# Usage:
#   bash scripts/12_build_lm_from_hf.sh
#   bash scripts/12_build_lm_from_hf.sh --order 5    # for a 5-gram LM
#   bash scripts/12_build_lm_from_hf.sh --skip-download  # if corpora already cached
#
# Output:
#   data/lm/lt_token_4gram_v2.arpa  (or v2_5gram.arpa for --order 5)
#
# No HF auth needed — the dataset repo is public.

set -euo pipefail

REPO_ID="sliderforthewin/lt-asr-lm-corpora"
CORPORA_DIR="data/lm/hf-corpora"
ORDER=4
MIN_COUNT=2
SKIP_DOWNLOAD=false

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --order) ORDER="$2"; shift 2 ;;
        --min-count) MIN_COUNT="$2"; shift 2 ;;
        --skip-download) SKIP_DOWNLOAD=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

OUTNAME="lt_token_${ORDER}gram_v2.arpa"
OUT="data/lm/${OUTNAME}"
COMBINED="data/lm/lt_combined_hf.txt"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo "=== Build parakeet token-level ${ORDER}-gram LM from HF corpora ==="
echo "  repo:   ${REPO_ID}"
echo "  output: ${OUT}"
echo ""

# Step 1: Download corpora from HF (no auth needed)
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo "--- Step 1: Downloading corpora from HuggingFace ---"
    mkdir -p "$CORPORA_DIR"

    # Try 'hf' CLI first, fall back to 'huggingface-cli'
    if command -v hf &>/dev/null; then
        DL_CMD="hf download"
    elif command -v huggingface-cli &>/dev/null; then
        DL_CMD="huggingface-cli download"
    else
        echo "ERROR: Neither 'hf' nor 'huggingface-cli' found. Install: pip install huggingface_hub"
        exit 1
    fi

    $DL_CMD "$REPO_ID" \
        --repo-type dataset \
        --include 'corpora/*.zst' \
        --local-dir "$CORPORA_DIR"

    echo "  Downloaded to ${CORPORA_DIR}/corpora/"
else
    echo "--- Step 1: Skipping download (--skip-download) ---"
    if [ ! -d "${CORPORA_DIR}/corpora" ]; then
        echo "ERROR: ${CORPORA_DIR}/corpora/ not found. Run without --skip-download first."
        exit 1
    fi
fi
echo ""

# Step 2: Decompress + concatenate all sources into one plain-text file
echo "--- Step 2: Decompressing and concatenating corpora ---"

# Check for zstdcat
if ! command -v zstdcat &>/dev/null; then
    if command -v zstd &>/dev/null; then
        ZSTDCAT="zstd -dc"
    else
        echo "ERROR: zstdcat/zstd not found. Install: brew install zstd (or apt install zstd)"
        exit 1
    fi
else
    ZSTDCAT="zstdcat"
fi

echo "  Combining into ${COMBINED}..."
> "$COMBINED"  # truncate
for f in "${CORPORA_DIR}"/corpora/*.sentences.txt.zst; do
    if [ -f "$f" ]; then
        name="$(basename "$f")"
        echo -n "    +${name}... "
        lines_before=$(wc -l < "$COMBINED")
        $ZSTDCAT "$f" >> "$COMBINED"
        lines_after=$(wc -l < "$COMBINED")
        added=$((lines_after - lines_before))
        echo "${added} lines"
    fi
done
total=$(wc -l < "$COMBINED")
size_mb=$(du -m "$COMBINED" | cut -f1)
echo "  Total: ${total} lines, ${size_mb} MB"
echo ""

# Step 3: Build the token-level LM
echo "--- Step 3: Building token-level ${ORDER}-gram LM ---"
echo "  This tokenizes all ${total} sentences with parakeet's BPE and counts"
echo "  subword n-grams. May take 10-30 min for ~60M sentences."
echo ""

# Include the training manifests too (exact domain match, highest per-token value)
MANIFEST_ARGS=""
for m in data/manifests/*.json; do
    if [ -f "$m" ]; then
        MANIFEST_ARGS="${MANIFEST_ARGS} --manifests ${m}"
    fi
done

python scripts/08b_build_token_lm.py \
    ${MANIFEST_ARGS} \
    --text-corpus "$COMBINED" \
    --order "$ORDER" \
    --min-count "$MIN_COUNT" \
    --out "$OUT"

echo ""
echo "=== Done ==="
echo "  LM file: ${OUT}"
echo "  Size: $(du -h "$OUT" | cut -f1)"
echo ""
echo "  To use in inference:"
echo "    cfg.beam.ngram_lm_model = '${OUT}'"
echo "    cfg.beam.ngram_lm_alpha = 0.5"
echo "    model.change_decoding_strategy(cfg)"
