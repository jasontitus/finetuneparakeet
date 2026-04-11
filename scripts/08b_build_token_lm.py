#!/usr/bin/env python3
"""Build a SUBWORD-TOKEN n-gram LM for NeMo TDT beam decoding.

NeMo's beam decoder queries the LM with `chr(token_id + DEFAULT_TOKEN_OFFSET)`
at each step, NOT with words. An LM trained on actual word text will never
match and silently do nothing (or hurt when alpha is large and the "unknown"
backoff probabilities dominate).

This script:
1. Loads the parakeet tokenizer
2. Tokenizes each input sentence into subword IDs
3. Maps each ID to chr(id + 100) to produce a sentence of single-character
   "tokens"
4. Counts n-grams over these sequences
5. Writes ARPA format suitable for NeMo's TDT beam decoder

Run:
    python scripts/08b_build_token_lm.py \\
        --manifests data/manifests/cv25_lt_train.json \\
                    data/manifests/cv25_lt_dev.json \\
                    data/manifests/voxpopuli_lt_train.json \\
                    data/manifests/fleurs_lt_train.json \\
                    data/manifests/shunyalabs_lt_train.json \\
        --text-corpus data/lm/lt_wikipedia.txt \\
        --order 4 --min-count 2 \\
        --out data/lm/lt_token_4gram.arpa
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"

DEFAULT_TOKEN_OFFSET = 100  # must match nemo.collections.asr.parts.submodules.ngram_lm.constants


def read_sentences(manifests: list[Path], text_corpora: list[Path]) -> list[str]:
    """Return plain-text sentences from manifests + text corpora."""
    sents: list[str] = []
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
                t = (d.get("text") or d.get("transcript") or "").strip()
                if t:
                    sents.append(t)
                    n += 1
        print(f"  {m.name}: {n:,} sentences", flush=True)
    for tc in text_corpora:
        if not tc.exists():
            print(f"  ! missing corpus: {tc}", file=sys.stderr)
            continue
        n = 0
        with tc.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    sents.append(line)
                    n += 1
        print(f"  {tc.name}: {n:,} sentences", flush=True)
    return sents


def tokenize_all(sents: list[str], tokenizer, batch_size: int = 1000) -> list[list[str]]:
    """Tokenize each sentence to list of chr(id + offset) tokens."""
    import string
    out: list[list[str]] = []
    for i, s in enumerate(sents):
        ids = tokenizer.text_to_ids(s)
        chars = [chr(tid + DEFAULT_TOKEN_OFFSET) for tid in ids]
        out.append(chars)
        if (i + 1) % 100000 == 0:
            print(f"    tokenized {i+1:,}/{len(sents):,}", flush=True)
    return out


def count_ngrams(sequences: list[list[str]], order: int) -> tuple[list[Counter], set[str]]:
    counters = [Counter() for _ in range(order)]
    vocab: set[str] = {BOS, EOS, UNK}
    for tokens in sequences:
        full = [BOS] + tokens + [EOS]
        vocab.update(full)
        for n in range(1, order + 1):
            counters[n - 1].update(
                tuple(full[i : i + n]) for i in range(len(full) - n + 1)
            )
    return counters, vocab


def write_arpa(path: Path, counters: list[Counter], vocab: set[str], order: int, bow: float = -0.4):
    V = len(vocab)
    total_unigram = sum(counters[0].values())
    unigram_logp: dict[tuple[str, ...], float] = {}
    for tok in vocab:
        count = counters[0].get((tok,), 0)
        p = (count + 1) / (total_unigram + V)
        unigram_logp[(tok,)] = math.log10(p)

    higher_logp: list[dict[tuple[str, ...], float]] = [{} for _ in range(order - 1)]
    for n in range(2, order + 1):
        ctx_counts = counters[n - 2]
        for ngram, c in counters[n - 1].items():
            denom = ctx_counts.get(ngram[:-1], 0)
            if denom == 0:
                continue
            higher_logp[n - 2][ngram] = math.log10(c / denom)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\\data\\\n")
        f.write(f"ngram 1={len(unigram_logp)}\n")
        for n in range(2, order + 1):
            f.write(f"ngram {n}={len(higher_logp[n - 2])}\n")
        f.write("\n")

        f.write("\\1-grams:\n")
        for ng, lp in unigram_logp.items():
            tok = ng[0]
            if order > 1:
                f.write(f"{lp:.6f}\t{tok}\t{bow:.6f}\n")
            else:
                f.write(f"{lp:.6f}\t{tok}\n")
        f.write("\n")

        for n in range(2, order + 1):
            f.write(f"\\{n}-grams:\n")
            for ng, lp in higher_logp[n - 2].items():
                ng_str = " ".join(ng)
                if n < order:
                    f.write(f"{lp:.6f}\t{ng_str}\t{bow:.6f}\n")
                else:
                    f.write(f"{lp:.6f}\t{ng_str}\n")
            f.write("\n")
        f.write("\\end\\\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3")
    ap.add_argument("--manifests", type=Path, nargs="*", default=[])
    ap.add_argument("--text-corpus", type=Path, nargs="*", default=[])
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if not args.manifests and not args.text_corpus:
        print("error: need --manifests and/or --text-corpus", file=sys.stderr)
        return 2

    import warnings; warnings.filterwarnings("ignore")
    import logging; logging.disable(logging.WARNING)
    import nemo.collections.asr as nemo_asr

    print(f"▸ loading tokenizer from {args.model}...", flush=True)
    model = nemo_asr.models.ASRModel.from_pretrained(args.model, map_location="cpu")
    tokenizer = model.tokenizer
    print(f"  vocab_size: {tokenizer.vocab_size}", flush=True)
    # Free the model — we only need the tokenizer
    del model

    print(f"▸ reading corpora...", flush=True)
    sents = read_sentences(args.manifests, args.text_corpus)
    print(f"  total: {len(sents):,} sentences", flush=True)

    print(f"▸ tokenizing into subword IDs (offset={DEFAULT_TOKEN_OFFSET})...", flush=True)
    token_seqs = tokenize_all(sents, tokenizer)
    total_tokens = sum(len(s) for s in token_seqs)
    print(f"  {total_tokens:,} total tokens ({total_tokens/len(sents):.1f}/sentence avg)", flush=True)

    print(f"▸ counting {args.order}-grams...", flush=True)
    counters, vocab = count_ngrams(token_seqs, args.order)
    for n, c in enumerate(counters, start=1):
        print(f"  {n}-grams: {len(c):,}  (total counts: {sum(c.values()):,})", flush=True)
    print(f"  vocab: {len(vocab):,}", flush=True)

    if args.min_count > 1:
        for i in range(1, args.order):
            before = len(counters[i])
            counters[i] = Counter({ng: c for ng, c in counters[i].items() if c >= args.min_count})
            print(f"  filtered {i+1}-grams: {before:,} → {len(counters[i]):,} (min_count={args.min_count})", flush=True)

    print(f"▸ writing ARPA to {args.out}...", flush=True)
    write_arpa(args.out, counters, vocab, args.order)
    size_mb = args.out.stat().st_size / 1e6
    print(f"▸ done. {args.out} ({size_mb:.1f} MB)", flush=True)

    # Sanity check: load in kenlm and score
    try:
        import kenlm
        m = kenlm.Model(str(args.out))
        # Score a tokenized LT sentence
        test = "lietuvos sostinė yra vilnius"
        ids = tokenizer.text_to_ids(test)
        chars = [chr(tid + DEFAULT_TOKEN_OFFSET) for tid in ids]
        score = m.score(" ".join(chars), bos=True, eos=True)
        print(f"▸ kenlm sanity: '{test}' → {len(ids)} tokens, score={score:.3f}", flush=True)
    except Exception as e:
        print(f"▸ kenlm sanity failed: {e}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
