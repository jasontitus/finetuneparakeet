#!/usr/bin/env python3
"""Audit the eval normalizer to find WER that isn't really ASR error.

Looks for patterns in baseline failures that are annotation/
normalization noise, not acoustic model mistakes:

1. Numbers ref-as-digits vs hyp-as-words (or vice versa)
2. Abbreviation expansion (e.g. "dr." vs "daktaras")
3. Unicode/diacritic normalization mismatches
4. Hyphen/compound-word splitting
5. Case-different-after-normalization (shouldn't happen but verify)
6. Punctuation-only differences that normalization didn't strip

Run:
    python scripts/07_eval_audit.py results/baseline_test_full/per_clip.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


# Number → Lithuanian word mapping (spot-check only; not exhaustive).
LT_NUMBER_WORDS = {
    "0": "nulis", "1": "vienas", "2": "du", "3": "trys", "4": "keturi",
    "5": "penki", "6": "šeši", "7": "septyni", "8": "aštuoni", "9": "devyni",
    "10": "dešimt", "11": "vienuolika", "12": "dvylika", "13": "trylika",
    "20": "dvidešimt", "100": "šimtas", "1000": "tūkstantis",
}

ABBREVIATIONS = {
    "dr.": "daktaras",
    "prof.": "profesorius",
    "a.": "amžiuje",
    "pvz.": "pavyzdžiui",
    "tt.": "taip toliau",
    "kt.": "kiti",
    "ir t.t.": "ir taip toliau",
    "psl.": "puslapis",
    "str.": "straipsnis",
}


def has_digit(s: str) -> bool:
    return any(c.isdigit() for c in s)


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def nfd(s: str) -> str:
    return unicodedata.normalize("NFD", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("per_clip_jsonl", type=Path)
    args = ap.parse_args()

    clips: list[dict] = []
    with args.per_clip_jsonl.open() as f:
        for line in f:
            if line.strip():
                clips.append(json.loads(line))

    print(f"▸ loaded {len(clips):,} clips", flush=True)

    # --- Check 1: digit vs word mismatches in REF vs HYP ---
    print("\n== Digit-related mismatches ==")
    digit_ref_clips = [c for c in clips if has_digit(c["reference_raw"])]
    digit_hyp_clips = [c for c in clips if has_digit(c["hypothesis_raw"])]
    print(f"  clips with digits in REF: {len(digit_ref_clips)}")
    print(f"  clips with digits in HYP: {len(digit_hyp_clips)}")
    # The earlier error analysis found 0 digit-related errors, but let's
    # see if the REF has digits that the model should read as words.
    for c in digit_ref_clips[:10]:
        print(f"  REF: {c['reference_raw'][:80]}")
        print(f"  HYP: {c['hypothesis_raw'][:80]}")
        print()

    # --- Check 2: abbreviation mismatches ---
    print("\n== Abbreviation mismatches ==")
    abbr_clips = []
    for c in clips:
        ref = c["reference_raw"].lower()
        if any(a in ref for a in ABBREVIATIONS):
            abbr_clips.append(c)
    print(f"  clips with abbreviations in REF: {len(abbr_clips)}")
    for c in abbr_clips[:5]:
        print(f"  REF: {c['reference_raw'][:80]}")
        print(f"  HYP: {c['hypothesis_raw'][:80]}")
        print()

    # --- Check 3: NFC vs NFD mismatches ---
    print("\n== Unicode normalization check ==")
    nfc_mismatch = 0
    for c in clips:
        if nfc(c["reference_raw"]) != c["reference_raw"]:
            nfc_mismatch += 1
    print(f"  refs not already in NFC: {nfc_mismatch}")
    # Our normalizer applies NFC, so this shouldn't cause issues,
    # but worth knowing the raw data quality.

    # --- Check 4: Hyphen / compound words ---
    print("\n== Hyphen errors ==")
    hyphen_errs = 0
    for c in clips:
        if "-" in c["reference_raw"] and "-" not in c["hypothesis_raw"]:
            hyphen_errs += 1
    print(f"  clips with hyphen in REF but not HYP: {hyphen_errs}")

    # --- Check 5: Short clip quality ---
    print("\n== Short-clip analysis (<3s) ==")
    short_clips = [c for c in clips if c["duration"] < 3.0]
    print(f"  short clips: {len(short_clips)}")
    short_perfect = sum(1 for c in short_clips
                       if c.get("reference_norm") == c.get("hypothesis_norm"))
    print(f"  perfect among short: {short_perfect} ({100*short_perfect/max(len(short_clips),1):.1f}%)")
    # Show some short failures
    short_fails = [c for c in short_clips
                   if c.get("reference_norm") != c.get("hypothesis_norm")][:5]
    for c in short_fails:
        print(f"  [{c['duration']:.1f}s]")
        print(f"    REF: {c['reference_raw'][:80]}")
        print(f"    HYP: {c['hypothesis_raw'][:80]}")

    # --- Check 6: Catastrophic clips (WER > 100%) ---
    # These are likely the most impactful on overall WER.
    print("\n== Catastrophic clips (drift to wrong language, etc) ==")
    import jiwer
    cats = []
    for c in clips:
        ref = c.get("reference_norm") or c["reference_raw"].lower()
        hyp = c.get("hypothesis_norm") or c["hypothesis_raw"].lower()
        if ref == hyp:
            continue
        try:
            wer = jiwer.wer(ref, hyp) * 100
        except Exception:
            continue
        if wer >= 100:
            cats.append((wer, c))
    print(f"  clips with >=100% WER: {len(cats)}")
    # Show language the hypothesis drifted into
    for wer, c in cats[:15]:
        hyp = c["hypothesis_raw"][:60]
        print(f"  [{wer:5.0f}%] {c['duration']:4.1f}s  HYP: {hyp}")

    # --- Check 7: What fraction of TOTAL errors come from top-100 worst clips?
    print("\n== Error concentration ==")
    all_errors = []
    for c in clips:
        ref = c.get("reference_norm") or c["reference_raw"].lower()
        hyp = c.get("hypothesis_norm") or c["hypothesis_raw"].lower()
        if ref == hyp:
            all_errors.append((0, c, 0, 0))
            continue
        try:
            out = jiwer.process_words(ref, hyp)
            n_errs = out.substitutions + out.insertions + out.deletions
            n_words = len(ref.split())
            all_errors.append((n_errs, c, n_words, n_errs / max(n_words, 1) * 100))
        except Exception:
            continue
    total_errs = sum(e[0] for e in all_errors)
    total_words = sum(e[2] for e in all_errors)
    print(f"  total word errors:      {total_errs:,}")
    print(f"  total reference words:  {total_words:,}")
    print(f"  corpus-level WER:       {100*total_errs/total_words:.2f}%")
    # Sort by errors
    all_errors.sort(key=lambda x: -x[0])
    for top_k in [10, 50, 100, 500]:
        top_errs = sum(e[0] for e in all_errors[:top_k])
        print(f"  errors from top {top_k:4d} clips:  {top_errs:,}  ({100*top_errs/total_errs:.1f}% of total)")


if __name__ == "__main__":
    main()
