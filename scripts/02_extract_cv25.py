#!/usr/bin/env python3
"""Extract CV25 LT tarball (if needed) and print split stats.

Idempotent — safe to re-run. Walks the extracted corpus, reads the
TSVs, joins against `clip_durations.tsv` to compute wall-clock hours
per split, and writes a tiny JSON summary to
`data/cv25_lt/splits_summary.json`.

Run:
    python scripts/02_extract_cv25.py \
        [--tarball ~/Downloads/...lt.tar.gz] \
        [--dest data/cv25_lt]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_TARBALL = Path.home() / "Downloads" / "1774126516509-cv-corpus-25.0-2026-03-09-lt.tar.gz"
DEFAULT_DEST = Path(__file__).resolve().parent.parent / "data" / "cv25_lt"


def find_lang_dir(dest: Path) -> Path:
    """Locate the <corpus>/lt/ directory inside `dest`. Extracts if missing."""
    # Look for any */lt/test.tsv
    for p in dest.rglob("lt/test.tsv"):
        return p.parent
    return None  # type: ignore[return-value]


def extract(tarball: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"▸ extracting {tarball} → {dest}", flush=True)
    # Use tar directly — faster than python's tarfile for large tarballs.
    subprocess.run(
        ["tar", "-xzf", str(tarball), "-C", str(dest)],
        check=True,
    )


def read_clip_durations(path: Path) -> dict[str, float]:
    """clip_durations.tsv has columns: clip\tduration[ms]."""
    out: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        if header[0].strip().lower() != "clip":
            raise RuntimeError(f"unexpected header in {path}: {header}")
        for row in reader:
            if len(row) < 2:
                continue
            clip = row[0].strip()
            try:
                ms = float(row[1])
            except ValueError:
                continue
            out[clip] = ms / 1000.0
    return out


def tsv_clip_set(path: Path) -> set[str]:
    out: set[str] = set()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            p = (row.get("path") or "").strip()
            if p:
                out.add(p)
    return out


def summarize(lang_dir: Path) -> dict:
    durations = read_clip_durations(lang_dir / "clip_durations.tsv")
    print(f"  clip_durations.tsv: {len(durations):,} clips", flush=True)

    splits = ["train", "dev", "test", "validated", "other", "invalidated"]
    summary: dict = {"corpus_dir": str(lang_dir), "n_clips_total": len(durations), "splits": {}}

    clip_sets = {s: tsv_clip_set(lang_dir / f"{s}.tsv") for s in splits if (lang_dir / f"{s}.tsv").exists()}

    for s, clips in clip_sets.items():
        dur = sum(durations.get(c, 0.0) for c in clips)
        matched = sum(1 for c in clips if c in durations)
        summary["splits"][s] = {
            "n_clips": len(clips),
            "n_with_duration": matched,
            "hours": round(dur / 3600.0, 3),
        }
        print(
            f"  {s:<13} {len(clips):>6,} clips  "
            f"{dur/3600:>6.2f} h  (matched {matched:,})",
            flush=True,
        )

    # validated_extra = validated − (train ∪ dev ∪ test)
    if "validated" in clip_sets and "train" in clip_sets:
        used = clip_sets.get("train", set()) | clip_sets.get("dev", set()) | clip_sets.get("test", set())
        extra = clip_sets["validated"] - used
        dur = sum(durations.get(c, 0.0) for c in extra)
        summary["splits"]["validated_extra"] = {
            "n_clips": len(extra),
            "hours": round(dur / 3600.0, 3),
            "note": "validated − (train ∪ dev ∪ test)",
        }
        print(
            f"  {'validated_ex':<13} {len(extra):>6,} clips  "
            f"{dur/3600:>6.2f} h  (validated minus train/dev/test)",
            flush=True,
        )

    # Target training bundle: train ∪ validated_extra ∪ other.
    train_bundle = (
        clip_sets.get("train", set())
        | (clip_sets.get("validated", set()) - clip_sets.get("test", set()) - clip_sets.get("dev", set()))
        | clip_sets.get("other", set())
    )
    # Make sure test never leaks in.
    train_bundle -= clip_sets.get("test", set())
    dur = sum(durations.get(c, 0.0) for c in train_bundle)
    summary["train_bundle"] = {
        "n_clips": len(train_bundle),
        "hours": round(dur / 3600.0, 3),
        "definition": "(train ∪ validated ∪ other) − (dev ∪ test)",
    }
    print(
        f"\n▸ training bundle  {len(train_bundle):,} clips  {dur/3600:.2f} h  "
        f"(train ∪ validated ∪ other − dev − test)",
        flush=True,
    )

    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarball", type=Path, default=DEFAULT_TARBALL)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = ap.parse_args()

    dest = args.dest
    lang_dir = find_lang_dir(dest)
    if lang_dir is None:
        if not args.tarball.exists():
            print(f"error: tarball not found: {args.tarball}", file=sys.stderr)
            return 2
        extract(args.tarball, dest)
        lang_dir = find_lang_dir(dest)
        if lang_dir is None:
            print("error: extraction did not produce lt/test.tsv", file=sys.stderr)
            return 3
    else:
        print(f"▸ corpus already extracted at {lang_dir}", flush=True)

    summary = summarize(lang_dir)

    out_path = dest / "splits_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n▸ summary → {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
