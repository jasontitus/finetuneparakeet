#!/usr/bin/env python3
"""Stream-tokenize text into chr(id+offset) format for KenLM lmplz.

Reads sentences from --text-corpus files and/or --manifests (NeMo JSON-lines),
tokenizes each with parakeet's BPE tokenizer, and writes space-separated
chr(id + 100) tokens to stdout — one line per sentence.

Designed for piping directly into lmplz:

    python scripts/13_stream_tokenize.py \\
        --text-corpus data/lm/lt_combined_hf.txt \\
        --manifests data/manifests/*.json \\
    | lmplz --order 4 -S 50% --prune 0 2 2 2 > model.arpa

Memory: ~2.5 GB briefly to load the ASR model for its tokenizer, then
~200 MB steady-state. No sentence lists or token arrays are accumulated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_TOKEN_OFFSET = 100  # must match nemo ngram_lm constants


def stream_sentences(manifests: list[Path], text_corpora: list[Path]):
    """Yield plain-text sentences one at a time — no list accumulation."""
    for m in manifests:
        if not m.exists():
            print(f"  ! missing manifest: {m}", file=sys.stderr)
            continue
        n = 0
        with m.open() as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                t = (d.get("text") or d.get("transcript") or "").strip()
                if t:
                    yield t
                    n += 1
        print(f"  manifest {m.name}: {n:,} sentences", file=sys.stderr, flush=True)

    for tc in text_corpora:
        if not tc.exists():
            print(f"  ! missing corpus: {tc}", file=sys.stderr)
            continue
        n = 0
        with tc.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line
                    n += 1
        print(f"  corpus {tc.name}: {n:,} sentences", file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="Stream-tokenize text for KenLM lmplz (stdout)")
    ap.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3",
                    help="NeMo ASR model (used only for its tokenizer)")
    ap.add_argument("--manifests", type=Path, nargs="*", default=[])
    ap.add_argument("--text-corpus", type=Path, nargs="*", default=[])
    args = ap.parse_args()

    if not args.manifests and not args.text_corpus:
        print("error: need --manifests and/or --text-corpus", file=sys.stderr)
        return 2

    import warnings; warnings.filterwarnings("ignore")
    import logging; logging.disable(logging.WARNING)
    import nemo.collections.asr as nemo_asr

    print(f"▸ loading tokenizer from {args.model}...", file=sys.stderr, flush=True)
    model = nemo_asr.models.ASRModel.from_pretrained(args.model, map_location="cpu")
    tokenizer = model.tokenizer
    print(f"  vocab_size: {tokenizer.vocab_size}", file=sys.stderr, flush=True)
    del model
    import gc; gc.collect()

    print("▸ streaming tokenized sentences to stdout...", file=sys.stderr, flush=True)
    write = sys.stdout.write
    count = 0
    for sent in stream_sentences(args.manifests, args.text_corpus):
        ids = tokenizer.text_to_ids(sent)
        if ids:
            write(" ".join(chr(tid + DEFAULT_TOKEN_OFFSET) for tid in ids))
            write("\n")
            count += 1
            if count % 1_000_000 == 0:
                print(f"    {count:,} sentences", file=sys.stderr, flush=True)

    print(f"▸ done: {count:,} sentences tokenized", file=sys.stderr, flush=True)


if __name__ == "__main__":
    sys.exit(main())
