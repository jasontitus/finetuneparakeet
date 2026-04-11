#!/usr/bin/env python3
"""Error analysis on per-clip eval output.

Takes a `per_clip.jsonl` from 04_eval.py and categorizes the errors
to understand WHERE the model fails. This tells us if fine-tuning
can even help, or if the ceiling is elsewhere (tokenizer, LM,
transcription inconsistencies).

Categories analyzed:
- Clips with zero errors vs errors
- Error type breakdown (insertions / deletions / substitutions)
- Morphological endings (Lithuanian noun/verb inflections)
- Named entities / capitalized words
- Number / digit errors
- Foreign-looking words (Latin letters in Lithuanian context)
- Short vs long clips
- WER distribution (how many clips have >50% WER?)

Run:
    python scripts/06_error_analysis.py results/baseline_test_full/per_clip.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import jiwer  # type: ignore

LT_LETTERS = set("aąbcčdeęėfghiįyjklmnoprsštuųūvzž")
# Foreign letters (Latin alphabet letters not in Lithuanian)
FOREIGN_LETTERS = set("qwx")

# Common LT suffixes - morphological markers that often change WER
NOUN_ENDINGS = ["as", "is", "us", "ys", "ai", "iai", "ui", "iui", "ą", "į", "ų", "ų",
                "e", "ę", "os", "ės", "ių", "ams", "iems", "oms", "ėms",
                "ų", "ą", "ose", "ėse", "uose", "iuose"]
VERB_ENDINGS = ["ti", "au", "ei", "ė", "ome", "ote", "o", "si", "siu", "sis",
                "siu", "sim", "site", "s", "damas", "damasi", "damasis", "tis"]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower().strip()
    return re.sub(r"\s+", " ", s)


def word_errors(ref: str, hyp: str) -> tuple[int, int, int]:
    """Return (substitutions, insertions, deletions)."""
    out = jiwer.process_words(ref, hyp)
    return out.substitutions, out.insertions, out.deletions


def get_diff_words(ref: str, hyp: str) -> list[tuple[str, str]]:
    """Return list of (ref_word, hyp_word) pairs where they differ."""
    ref_words = ref.split()
    hyp_words = hyp.split()
    # Use jiwer to get aligned edit operations
    out = jiwer.process_words(ref, hyp)
    pairs: list[tuple[str, str]] = []
    for chunks in out.alignments:
        for c in chunks:
            if c.type == "equal":
                continue
            r = " ".join(ref_words[c.ref_start_idx:c.ref_end_idx])
            h = " ".join(hyp_words[c.hyp_start_idx:c.hyp_end_idx])
            pairs.append((r, h))
    return pairs


def has_digit(s: str) -> bool:
    return any(c.isdigit() for c in s)


def has_foreign(s: str) -> bool:
    lower = s.lower()
    return any(c in FOREIGN_LETTERS for c in lower if c.isalpha())


def shared_prefix(a: str, b: str) -> int:
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i


def is_ending_only_error(ref: str, hyp: str) -> bool:
    """True if ref and hyp share a common prefix and differ only in the suffix."""
    if not ref or not hyp:
        return False
    pfx = shared_prefix(ref, hyp)
    return pfx >= 3 and (len(ref) - pfx <= 4 or len(hyp) - pfx <= 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("per_clip_jsonl", type=Path)
    ap.add_argument("--compare-to", type=Path, default=None,
                    help="Second per_clip.jsonl (e.g. finetuned) to diff against")
    args = ap.parse_args()

    clips: list[dict] = []
    with args.per_clip_jsonl.open() as f:
        for line in f:
            if line.strip():
                clips.append(json.loads(line))

    n = len(clips)
    print(f"▸ loaded {n:,} clips from {args.per_clip_jsonl}", flush=True)

    # Per-clip WER
    perfect = 0
    clip_wers = []
    total_sub = total_ins = total_del = 0
    all_diff_pairs: list[tuple[str, str]] = []

    for c in clips:
        ref = norm(c.get("reference_norm") or c["reference_raw"])
        hyp = norm(c.get("hypothesis_norm") or c["hypothesis_raw"])
        if ref == hyp:
            perfect += 1
            clip_wers.append(0.0)
            continue
        wer = jiwer.wer(ref, hyp) * 100
        clip_wers.append(wer)

        s, i, d = word_errors(ref, hyp)
        total_sub += s
        total_ins += i
        total_del += d
        all_diff_pairs.extend(get_diff_words(ref, hyp))

    print(f"\n== Per-clip WER distribution ==")
    print(f"  perfect (0% WER):  {perfect:,}  ({100*perfect/n:.1f}%)")
    buckets = [0.01, 5, 10, 20, 50, 100, 101]
    counts = [0] * (len(buckets) - 1)
    for w in clip_wers:
        for b in range(len(buckets) - 1):
            if buckets[b] <= w < buckets[b + 1] or (b == len(buckets) - 2 and w >= buckets[b]):
                counts[b] += 1
                break
    labels = ["0-5%", "5-10%", "10-20%", "20-50%", "50-100%", ">=100%"]
    for l, c in zip(labels, counts):
        print(f"  {l:>10}:  {c:,}  ({100*c/n:.1f}%)")

    print(f"\n== Error type breakdown ==")
    total_errs = total_sub + total_ins + total_del
    if total_errs > 0:
        print(f"  substitutions:  {total_sub:,}  ({100*total_sub/total_errs:.1f}%)")
        print(f"  deletions:      {total_del:,}  ({100*total_del/total_errs:.1f}%)")
        print(f"  insertions:     {total_ins:,}  ({100*total_ins/total_errs:.1f}%)")

    print(f"\n== Top 40 substitution pairs ==")
    sub_counter = Counter()
    ending_errors = 0
    digit_errors = 0
    foreign_errors = 0
    for r, h in all_diff_pairs:
        if not r or not h:
            continue
        sub_counter[(r, h)] += 1
        if is_ending_only_error(r, h):
            ending_errors += 1
        if has_digit(r) or has_digit(h):
            digit_errors += 1
        if has_foreign(r) or has_foreign(h):
            foreign_errors += 1

    for (r, h), c in sub_counter.most_common(40):
        marker = ""
        if is_ending_only_error(r, h):
            marker = "  [ending]"
        elif has_digit(r) or has_digit(h):
            marker = "  [digit]"
        elif has_foreign(r) or has_foreign(h):
            marker = "  [foreign]"
        print(f"  {c:4d}× '{r}' → '{h}'{marker}")

    print(f"\n== Error category counts ==")
    n_diff = len(all_diff_pairs)
    print(f"  total diff pairs:  {n_diff:,}")
    print(f"  ending-only:       {ending_errors:,}  ({100*ending_errors/max(n_diff,1):.1f}%)")
    print(f"  digit-related:     {digit_errors:,}  ({100*digit_errors/max(n_diff,1):.1f}%)")
    print(f"  foreign-letter:    {foreign_errors:,}  ({100*foreign_errors/max(n_diff,1):.1f}%)")

    print(f"\n== Duration vs error rate ==")
    buckets = [(0, 3), (3, 6), (6, 10), (10, 15), (15, 30)]
    for lo, hi in buckets:
        ws = [w for c, w in zip(clips, clip_wers) if lo <= c["duration"] < hi]
        if ws:
            avg = sum(ws) / len(ws)
            print(f"  {lo:2d}-{hi:2d}s:  n={len(ws):4d}  avg_WER={avg:5.2f}%")

    # Show 5 high-error examples
    worst = sorted(
        [(w, c) for w, c in zip(clip_wers, clips) if w > 0],
        key=lambda x: -x[0],
    )[:5]
    print(f"\n== 5 highest-WER clips ==")
    for w, c in worst:
        print(f"  [{w:6.2f}%]")
        print(f"    REF: {c['reference_raw'][:100]}")
        print(f"    HYP: {c['hypothesis_raw'][:100]}")


if __name__ == "__main__":
    main()
