#!/usr/bin/env python3
"""Download and clean Lithuanian Wikipedia for LM training.

Uses HuggingFace datasets to pull the latest LT Wikipedia dump,
segments articles into sentences, normalizes text the same way as
our training manifests, and writes plain-text output.

Output is one sentence per line, normalized (NFC, lowercased,
punctuation stripped) to match how ASR hypotheses will be scored
by the LM at inference time.

Run:
    python scripts/10_download_lt_wikipedia.py \\
        --out data/lm/lt_wikipedia.txt

Then feed into scripts/08_build_lm.py:
    python scripts/08_build_lm.py \\
        --manifests data/manifests/ALL_train.json \\
                    data/manifests/cv25_lt_dev.json \\
        --text-corpus data/lm/lt_wikipedia.txt \\
        --order 4 --out data/lm/lt_wiki_4gram.arpa
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path


# Lithuanian sentence-ending heuristics. LT uses standard Latin
# punctuation so regular rules work.
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZĄČĘĖĮŠŲŪŽ])")
# Section headers, references, and infobox artifacts from Wikipedia.
BAD_PATTERNS = [
    re.compile(r"^\s*\d+\s*$"),          # line of just a number
    re.compile(r"^\s*[=\-*]+\s*$"),      # separator lines
    re.compile(r"^\s*\|"),                # table rows
    re.compile(r"\{\{.*?\}\}"),           # template remnants
    re.compile(r"^\s*[A-Z]:"),            # categorization like "File:"
]

PUNCT_RE = re.compile(r"[\"\'\,\.\!\?\;\:\(\)\[\]\{\}\«\»\"\"\„\‟\‚\'\'\–\—\-]")
SPACE_RE = re.compile(r"\s+")
DIGIT_RE = re.compile(r"\d")


def normalize(s: str) -> str:
    """Match the LM query-time normalization."""
    s = unicodedata.normalize("NFC", s)
    s = s.lower()
    s = PUNCT_RE.sub(" ", s)
    s = SPACE_RE.sub(" ", s).strip()
    return s


def clean_article(text: str) -> list[str]:
    """Turn a Wikipedia article into a list of normalized sentences."""
    sentences: list[str] = []
    # Split into paragraphs first to reduce cross-paragraph sentence joins
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph or len(paragraph) < 20:
            continue
        if any(p.search(paragraph) for p in BAD_PATTERNS):
            continue
        # Split paragraph into sentences
        for sent in SENT_SPLIT_RE.split(paragraph):
            sent = sent.strip()
            if len(sent) < 15:
                continue
            # Skip sentences with lots of digits (tables, dates lists)
            digit_frac = len(DIGIT_RE.findall(sent)) / max(len(sent), 1)
            if digit_frac > 0.1:
                continue
            # Normalize
            normed = normalize(sent)
            if not normed or len(normed.split()) < 3:
                continue
            # Filter: must be mostly Lithuanian-looking
            # (heuristic: at least one word containing a LT-specific letter)
            if not re.search(r"[ąčęėįšųūž]", normed):
                # Still keep if it's long enough to be a real sentence
                if len(normed.split()) < 5:
                    continue
            sentences.append(normed)
    return sentences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/lm/lt_wikipedia.txt"))
    ap.add_argument(
        "--config",
        default="20231101.lt",
        help="HF wikipedia config (e.g. 20231101.lt for Lithuanian)",
    )
    ap.add_argument("--max-articles", type=int, default=None,
                    help="Cap articles for a smoke test")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("error: datasets not installed", file=sys.stderr)
        return 1

    print(f"▸ loading wikimedia/wikipedia/{args.config}...", flush=True)
    try:
        ds = load_dataset("wikimedia/wikipedia", args.config, split="train")
    except Exception as e:
        print(f"  failed: {e}", flush=True)
        print(f"▸ falling back to legacy 'wikipedia/lt' loader...", flush=True)
        ds = load_dataset("wikipedia", language="lt", date="20231101",
                          split="train", trust_remote_code=True)

    total = len(ds)
    print(f"▸ {total:,} articles", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_sentences = 0
    n_articles = 0

    with args.out.open("w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if args.max_articles and i >= args.max_articles:
                break
            text = row.get("text", "")
            if not text:
                continue
            sents = clean_article(text)
            for s in sents:
                f.write(s + "\n")
                n_sentences += 1
            n_articles += 1
            if n_articles % 10000 == 0:
                print(f"  processed {n_articles:,} articles, {n_sentences:,} sentences", flush=True)

    size_mb = args.out.stat().st_size / 1e6
    print(f"\n▸ done. {n_articles:,} articles → {n_sentences:,} sentences", flush=True)
    print(f"  {args.out} ({size_mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
