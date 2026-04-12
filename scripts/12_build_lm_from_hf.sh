#!/usr/bin/env bash
# Build a token-level n-gram LM for parakeet from the lt-asr-lm HF corpora.
#
# Downloads Lithuanian text corpus from HuggingFace, stream-tokenizes with
# parakeet's BPE, and builds the ARPA file using KenLM's lmplz (disk-based
# merge-sort — bounded memory, handles any corpus size).
#
# Memory: ~4 GB (tokenizer) + lmplz sort buffers (configurable via --lmplz-mem).
# Works comfortably on 64 GB machines with 61M+ sentence corpora.
#
# Prerequisites:
#   pip install huggingface_hub zstandard nemo_toolkit[asr]
#   cmake + C++ compiler (for auto-building lmplz if not on PATH)
#     macOS: xcode-select --install && brew install cmake boost
#     Linux: apt install cmake g++ libboost-all-dev
#
# Usage:
#   bash scripts/12_build_lm_from_hf.sh
#   bash scripts/12_build_lm_from_hf.sh --order 5
#   bash scripts/12_build_lm_from_hf.sh --skip-download
#   bash scripts/12_build_lm_from_hf.sh --lmplz-mem 80%
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
LMPLZ_MEM="50%"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --order) ORDER="$2"; shift 2 ;;
        --min-count) MIN_COUNT="$2"; shift 2 ;;
        --skip-download) SKIP_DOWNLOAD=true; shift ;;
        --lmplz-mem) LMPLZ_MEM="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Build --prune args: no pruning for unigrams, MIN_COUNT for higher orders
PRUNE_ARGS="0"
for ((i=2; i<=ORDER; i++)); do
    PRUNE_ARGS="$PRUNE_ARGS $MIN_COUNT"
done

OUTNAME="lt_token_${ORDER}gram_v2.arpa"
OUT="data/lm/${OUTNAME}"
COMBINED="data/lm/lt_combined_hf.txt"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# --- Ensure lmplz is available (build KenLM from source if needed) ---
ensure_lmplz() {
    if command -v lmplz &>/dev/null; then
        LMPLZ="lmplz"
        return
    fi
    local build_dir="$PROJECT_DIR/tools/kenlm/build"
    if [ -x "$build_dir/bin/lmplz" ]; then
        LMPLZ="$build_dir/bin/lmplz"
        return
    fi
    echo "--- Building KenLM tools (one-time setup) ---"
    mkdir -p "$PROJECT_DIR/tools"
    if [ ! -d "$PROJECT_DIR/tools/kenlm" ]; then
        git clone --depth 1 https://github.com/kpu/kenlm.git "$PROJECT_DIR/tools/kenlm"
    fi
    mkdir -p "$build_dir"
    cmake -S "$PROJECT_DIR/tools/kenlm" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
    local njobs
    njobs=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    cmake --build "$build_dir" -j "$njobs" 2>&1 | tail -5
    if [ ! -x "$build_dir/bin/lmplz" ]; then
        echo "ERROR: Failed to build lmplz."
        echo "  macOS: xcode-select --install && brew install cmake boost"
        echo "  Linux: apt install cmake g++ libboost-all-dev"
        exit 1
    fi
    LMPLZ="$build_dir/bin/lmplz"
    echo "  Built: $LMPLZ"
}
ensure_lmplz

echo "=== Build parakeet token-level ${ORDER}-gram LM from HF corpora ==="
echo "  repo:      ${REPO_ID}"
echo "  output:    ${OUT}"
echo "  lmplz:     ${LMPLZ}"
echo "  lmplz mem: ${LMPLZ_MEM}"
echo "  prune:     ${PRUNE_ARGS}"
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

# Step 3: Stream-tokenize + build LM via lmplz
echo "--- Step 3: Stream-tokenize + build ${ORDER}-gram LM via lmplz ---"
echo "  Piping ${total} sentences through parakeet BPE into lmplz."
echo "  Memory: ~4 GB (tokenizer) + ${LMPLZ_MEM} (lmplz sort buffers)"
echo ""

# Include training manifests (exact domain match — highest per-token value)
MANIFEST_ARGS=""
for m in data/manifests/*.json; do
    if [ -f "$m" ]; then
        MANIFEST_ARGS="${MANIFEST_ARGS} --manifests ${m}"
    fi
done

TMPDIR="data/lm/tmp"
mkdir -p "$TMPDIR" "$(dirname "$OUT")"

python scripts/13_stream_tokenize.py \
    ${MANIFEST_ARGS} \
    --text-corpus "$COMBINED" \
| "$LMPLZ" --order "$ORDER" \
    -S "$LMPLZ_MEM" \
    -T "$TMPDIR" \
    --prune $PRUNE_ARGS \
    --discount_fallback \
    > "$OUT"

rm -rf "$TMPDIR"

echo ""
echo "=== Done ==="
echo "  LM file: ${OUT}"
echo "  Size:    $(du -h "$OUT" | cut -f1)"

# Sanity check: verify kenlm can load the ARPA
python -c "
import kenlm
m = kenlm.Model('$OUT')
print(f'  kenlm load OK: order={m.order}')
" 2>/dev/null || echo "  (kenlm sanity check skipped — pip install kenlm to enable)"

echo ""
echo "  To use in inference:"
echo "    cfg.beam.ngram_lm_model = '${OUT}'"
echo "    cfg.beam.ngram_lm_alpha = 0.5"
echo "    model.change_decoding_strategy(cfg)"
