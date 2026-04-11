#!/usr/bin/env python3
"""Build a Lithuanian n-gram LM from our training manifests.

Pure Python implementation — writes out ARPA format that KenLM
Python bindings can load. Not as sophisticated as kenlm's lmplz
(no Kneser-Ney smoothing, just modified Witten-Bell), but enough
to get the rescoring infrastructure stood up.

Run:
    python scripts/08_build_lm.py \\
        --manifests data/manifests/cv25_lt_train.json \\
                    data/manifests/cv25_lt_dev.json \\
                    data/manifests/voxpopuli_lt_train.json \\
                    data/manifests/fleurs_lt_train.json \\
                    data/manifests/shunyalabs_lt_train.json \\
        --order 4 \\
        --out data/lm/lt_4gram.arpa

Then test with:
    python -c "import kenlm; m = kenlm.Model('data/lm/lt_4gram.arpa'); \\
        print(m.score('lietuva yra graži šalis', bos=True, eos=True))"
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"


def normalize(s: str) -> str:
    """Match how we'll present text to the LM at eval time."""
    s = unicodedata.normalize("NFC", s)
    s = s.lower()
    # Strip punctuation but keep Lithuanian diacritics
    s = re.sub(r"[\"\',\.\!\?\;\:\(\)\[\]\{\}\«\»\"\"\„\‟\‚\'\'\–\—\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def read_texts(manifests: list[Path]) -> list[str]:
    texts: list[str] = []
    for m in manifests:
        if not m.exists():
            print(f"  ! missing: {m}", file=sys.stderr)
            continue
        n = 0
        with m.open() as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                t = d.get("text") or d.get("transcript") or ""
                if not t:
                    continue
                t = normalize(t)
                if t:
                    texts.append(t)
                    n += 1
        print(f"  {m.name}: {n:,} sentences")
    return texts


def count_ngrams(texts: list[str], order: int):
    """Return list of ngram Counters, one per order from 1..order."""
    counters = [Counter() for _ in range(order)]
    vocab: set[str] = set()
    vocab.add(BOS)
    vocab.add(EOS)
    vocab.add(UNK)

    for text in texts:
        tokens = [BOS] + text.split() + [EOS]
        vocab.update(tokens)
        # For each n, count all n-grams in this sentence.
        for n in range(1, order + 1):
            counters[n - 1].update(
                tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
            )
    return counters, vocab


def write_arpa(path: Path, counters: list[Counter], vocab: set[str], order: int):
    """Write ARPA format with simple smoothing.

    We use a "stupid backoff" style where unigram probs are add-1
    smoothed and higher-order probs are MLE with a small backoff
    weight. This is much weaker than modified Kneser-Ney (what
    lmplz uses) but produces valid ARPA that kenlm can load.
    """
    V = len(vocab)
    total_unigram = sum(counters[0].values())
    # Unigram: add-1 smoothed log10 probability.
    unigram_logp: dict[tuple[str, ...], float] = {}
    for tok in vocab:
        count = counters[0].get((tok,), 0)
        p = (count + 1) / (total_unigram + V)
        unigram_logp[(tok,)] = math.log10(p)

    # Higher-order: MLE conditional probability.
    # P(w_n | w_{n-1..w_1}) = count(w_1..w_n) / count(w_1..w_{n-1})
    higher_logp: list[dict[tuple[str, ...], float]] = [{} for _ in range(order - 1)]
    for n in range(2, order + 1):
        idx = n - 1
        ctx_counts = counters[n - 2]
        for ngram, c in counters[n - 1].items():
            ctx = ngram[:-1]
            denom = ctx_counts.get(ctx, 0)
            if denom == 0:
                continue
            p = c / denom
            higher_logp[idx - 1][ngram] = math.log10(p)

    # Default backoff weight (log10).
    bow = math.log10(0.4)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\\data\\\n")
        f.write(f"ngram 1={len(unigram_logp)}\n")
        for n in range(2, order + 1):
            f.write(f"ngram {n}={len(higher_logp[n - 2])}\n")
        f.write("\n")

        # Unigrams
        f.write("\\1-grams:\n")
        for ng, lp in unigram_logp.items():
            tok = ng[0]
            if order > 1:
                f.write(f"{lp:.6f}\t{tok}\t{bow:.6f}\n")
            else:
                f.write(f"{lp:.6f}\t{tok}\n")
        f.write("\n")

        # Higher orders
        for n in range(2, order + 1):
            f.write(f"\\{n}-grams:\n")
            ngs = higher_logp[n - 2]
            for ng, lp in ngs.items():
                ngram_str = " ".join(ng)
                if n < order:
                    f.write(f"{lp:.6f}\t{ngram_str}\t{bow:.6f}\n")
                else:
                    f.write(f"{lp:.6f}\t{ngram_str}\n")
            f.write("\n")

        f.write("\\end\\\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", type=Path, nargs="+", required=True)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    print(f"▸ reading {len(args.manifests)} manifests...")
    texts = read_texts(args.manifests)
    print(f"▸ total sentences: {len(texts):,}")

    print(f"▸ counting {args.order}-grams...")
    counters, vocab = count_ngrams(texts, args.order)
    for n, c in enumerate(counters, start=1):
        print(f"  {n}-grams: {len(c):,}  (total counts: {sum(c.values()):,})")
    print(f"  vocab: {len(vocab):,}")

    print(f"▸ writing ARPA to {args.out}...")
    write_arpa(args.out, counters, vocab, args.order)
    size_mb = args.out.stat().st_size / 1e6
    print(f"▸ done. {args.out} ({size_mb:.1f} MB)")

    # Quick sanity check
    try:
        import kenlm
        m = kenlm.Model(str(args.out))
        s = "lietuva yra graži šalis"
        score = m.score(s, bos=True, eos=True)
        print(f"▸ kenlm loaded OK. score('{s}') = {score:.3f}")
    except Exception as e:
        print(f"▸ kenlm verification failed: {e}")


if __name__ == "__main__":
    main()
