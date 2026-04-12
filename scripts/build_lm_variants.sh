#!/bin/bash
# Build 3 LM variants for comparison.
# Run on Mac (CPU only, ~4-8 GB peak each).
set -euo pipefail
cd "$(dirname "$0")/.."

LMPLZ="${LMPLZ:-.venv/bin/lmplz}"
TOKENIZER="scripts/13_stream_tokenize.py"
MANIFESTS="data/manifests"
CORPORA="data/lm/hf-corpora/corpora"

echo "═══ Building 3 LM variants ═══"

# ── Variant 1: Old domain sources as 5-gram ──────────────────────
# Wikipedia + training manifests only (what made the old LM good, but now 5-gram)
echo ""
echo "▶ Variant 1: domain-only 5-gram (wiki + manifests)"
{
  zstd -dc "$CORPORA/wikipedia.sentences.txt.zst"
} > data/lm/lt_domain_only.txt
WC1=$(wc -l < data/lm/lt_domain_only.txt)
echo "  $WC1 sentences"

python "$TOKENIZER" \
  --manifests "$MANIFESTS"/cv25_lt_train.json "$MANIFESTS"/cv25_lt_dev.json \
              "$MANIFESTS"/fleurs_lt_train.json "$MANIFESTS"/voxpopuli_lt_train.json \
              "$MANIFESTS"/shunyalabs_lt_train.json \
  --text-corpus data/lm/lt_domain_only.txt \
| "$LMPLZ" -o 5 --discount_fallback -S 50% \
  > data/lm/lt_domain_5gram.arpa

echo "  → $(du -sh data/lm/lt_domain_5gram.arpa | cut -f1)"

# ── Variant 2: Wiki + OpenSubtitles (no CC-100 noise) ────────────
echo ""
echo "▶ Variant 2: wiki + subtitles 4-gram (no CC-100)"
{
  zstd -dc "$CORPORA/wikipedia.sentences.txt.zst"
  zstd -dc "$CORPORA/opensubtitles.sentences.txt.zst"
} > data/lm/lt_wiki_subs.txt
WC2=$(wc -l < data/lm/lt_wiki_subs.txt)
echo "  $WC2 sentences"

python "$TOKENIZER" \
  --manifests "$MANIFESTS"/cv25_lt_train.json "$MANIFESTS"/cv25_lt_dev.json \
              "$MANIFESTS"/fleurs_lt_train.json "$MANIFESTS"/voxpopuli_lt_train.json \
              "$MANIFESTS"/shunyalabs_lt_train.json \
  --text-corpus data/lm/lt_wiki_subs.txt \
| "$LMPLZ" -o 4 --discount_fallback -S 50% \
  > data/lm/lt_wikisubs_4gram.arpa

echo "  → $(du -sh data/lm/lt_wikisubs_4gram.arpa | cut -f1)"

# ── Variant 3: Interpolated (domain + broad) ─────────────────────
# Build the domain 4-gram first, then interpolate with the v2 broad 4-gram.
echo ""
echo "▶ Variant 3: interpolated (0.7 domain + 0.3 broad)"

# Domain 4-gram (same sources as variant 1 but 4-gram for matching order)
python "$TOKENIZER" \
  --manifests "$MANIFESTS"/cv25_lt_train.json "$MANIFESTS"/cv25_lt_dev.json \
              "$MANIFESTS"/fleurs_lt_train.json "$MANIFESTS"/voxpopuli_lt_train.json \
              "$MANIFESTS"/shunyalabs_lt_train.json \
  --text-corpus data/lm/lt_domain_only.txt \
| "$LMPLZ" -o 4 --discount_fallback -S 50% \
  > data/lm/lt_domain_4gram.arpa

echo "  domain: $(du -sh data/lm/lt_domain_4gram.arpa | cut -f1)"
echo "  broad:  $(du -sh data/lm/lt_token_4gram_v2.arpa | cut -f1)"

# Interpolate via Python
python3 << 'INTERP'
"""Simple ARPA linear interpolation: λ×P_a + (1-λ)×P_b for unigrams."""
import math
from collections import defaultdict

LAMBDA = 0.7  # domain weight
A = "data/lm/lt_domain_4gram.arpa"
B = "data/lm/lt_token_4gram_v2.arpa"
OUT = "data/lm/lt_interpolated_4gram.arpa"

def read_unigrams(path):
    probs = {}
    in_section = False
    with open(path) as f:
        for line in f:
            if line.strip() == "\\1-grams:":
                in_section = True; continue
            if line.strip().startswith("\\") and in_section:
                break
            if in_section and line.strip():
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    probs[parts[1]] = float(parts[0])
    return probs

print("  reading unigrams...")
ua = read_unigrams(A)
ub = read_unigrams(B)
all_tokens = set(ua) | set(ub)

# Default log prob for unseen tokens
default_lp = -7.0

interp = {}
for tok in all_tokens:
    pa = 10**ua.get(tok, default_lp)
    pb = 10**ub.get(tok, default_lp)
    combined = LAMBDA * pa + (1 - LAMBDA) * pb
    interp[tok] = math.log10(max(combined, 1e-10))

# Write interpolated unigrams into the DOMAIN arpa (keep higher-order n-grams from domain)
print("  writing interpolated ARPA...")
import shutil
shutil.copy2(A, OUT)

# Read the domain ARPA and replace unigram probs
with open(A) as f:
    lines = f.readlines()

with open(OUT, "w") as f:
    in_unigrams = False
    for line in lines:
        if line.strip() == "\\1-grams:":
            in_unigrams = True
            f.write(line)
            continue
        if line.strip().startswith("\\") and in_unigrams:
            in_unigrams = False
            f.write(line)
            continue
        if in_unigrams and line.strip():
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[1] in interp:
                parts[0] = f"{interp[parts[1]]:.6f}"
                f.write("\t".join(parts) + "\n")
                continue
        f.write(line)

print(f"  → {OUT}")
INTERP

echo "  → $(du -sh data/lm/lt_interpolated_4gram.arpa | cut -f1)"

# ── Summary ───────────────────────────────────────────────────────
echo ""
echo "═══ VARIANTS BUILT ═══"
for f in data/lm/lt_domain_5gram.arpa data/lm/lt_wikisubs_4gram.arpa data/lm/lt_interpolated_4gram.arpa; do
  [ -f "$f" ] && echo "  $(basename $f): $(du -sh $f | cut -f1)"
done
echo "▶ DONE"
