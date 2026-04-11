#!/usr/bin/env python3
"""Build NeMo-format JSON manifests for Lithuanian fine-tuning.

Expected to run on the L4 training VM (or locally, where available).

For each source dataset it:
  1. Downloads / pulls the data (HF for VoxPopuli/FLEURS/shunyalabs, local
     extracted tarball or GCS download for CV25 LT).
  2. Decodes audio to 16 kHz mono WAV on disk (so NeMo's loader can
     mmap cheaply and we don't pay MP3/FLAC decode cost on every epoch).
  3. Writes a NeMo-format manifest line: one JSON object per sample
     with `audio_filepath`, `duration`, and `text`.
  4. Keeps original text (mixed-case, with punctuation) to match the
     pretrained tokenizer's training format. Eval-time normalization
     (lowercase + strip punctuation) is applied equally to both
     reference and hypothesis in 04_eval.py so WER stays fair.

Output layout (all under `--out`, default `data/manifests`):

    manifests/
      cv25_lt_train.json      # CV25 LT train bundle (train ∪ validated ∪ other − dev − test)
      cv25_lt_dev.json        # CV25 LT dev (for during-training eval)
      cv25_lt_test.json       # CV25 LT test (held-out eval)
      voxpopuli_lt_train.json
      voxpopuli_lt_test.json
      fleurs_lt_train.json
      fleurs_lt_test.json
      shunyalabs_lt_train.json
      shunyalabs_lt_test.json
      ALL_train.json          # concatenation of all *_train.json (shuffled)
      audio/
        cv25_lt/<clip>.wav
        voxpopuli_lt/<id>.wav
        fleurs_lt/<id>.wav
        shunyalabs_lt/<id>.wav

Run with `--datasets cv25_lt` to start (fast smoke test). Add more as
you verify things work.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Callable, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "manifests"
DEFAULT_CV25_DIR = REPO_ROOT / "data" / "cv25_lt"
DEFAULT_CV25_GCS = "gs://safecare-maps-speechbench/corpora/cv25-lt/cv-corpus-25.0-2026-03-09-lt.tar.gz"

# Punctuation to strip. We KEEP Lithuanian letters and basic word chars.
PUNCT_RE = re.compile(r"[\"\'\,\.\!\?\;\:\(\)\[\]\{\}\«\»\“\”\„\‟\‚\‘\’\–\—\-]")
SPACE_RE = re.compile(r"\s+")


def clean_text(s: str) -> str:
    """NFC-normalize and collapse whitespace. Preserves case and punctuation
    so training targets match the pretrained tokenizer's expected format."""
    s = unicodedata.normalize("NFC", s)
    s = SPACE_RE.sub(" ", s).strip()
    return s


def write_wav_mono16k(audio_array, sample_rate: int, out_path: Path) -> float:
    """Write a float32 1-D numpy array as 16 kHz mono WAV. Returns duration in seconds."""
    import numpy as np
    import soundfile as sf

    arr = np.asarray(audio_array, dtype="float32")
    if arr.ndim == 2:
        arr = arr.mean(axis=1).astype("float32")
    if sample_rate != 16000:
        import librosa  # lazy — only needed for resample

        arr = librosa.resample(arr, orig_sr=sample_rate, target_sr=16000).astype("float32")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), arr, 16000, subtype="PCM_16")
    return float(len(arr) / 16000.0)


def decode_audio_file(path: Path) -> tuple["np.ndarray", int]:  # type: ignore[name-defined]
    """Best-effort decode of mp3/ogg/wav via soundfile, falling back to librosa."""
    import numpy as np
    import soundfile as sf

    try:
        data, sr = sf.read(str(path))
        return np.asarray(data, dtype="float32"), int(sr)
    except Exception:
        pass
    import librosa

    data, sr = librosa.load(str(path), sr=None, mono=True)
    return np.asarray(data, dtype="float32"), int(sr)


# ─── CV25 LT loaders ───────────────────────────────────────────────────────

def ensure_cv25(cv25_dir: Path, gcs_uri: str) -> Path:
    """Return the path to the <corpus>/lt/ directory, downloading+extracting if needed."""
    for p in cv25_dir.rglob("lt/test.tsv"):
        return p.parent
    cv25_dir.mkdir(parents=True, exist_ok=True)
    local_tar = cv25_dir / "cv25_lt.tar.gz"
    if not local_tar.exists():
        print(f"▸ fetching {gcs_uri} → {local_tar}", flush=True)
        subprocess.run(["gsutil", "-q", "cp", gcs_uri, str(local_tar)], check=True)
    print(f"▸ extracting {local_tar}", flush=True)
    subprocess.run(["tar", "-xzf", str(local_tar), "-C", str(cv25_dir)], check=True)
    try:
        local_tar.unlink()
    except OSError:
        pass
    for p in cv25_dir.rglob("lt/test.tsv"):
        return p.parent
    raise RuntimeError("cv25 lt/test.tsv not found after extraction")


def iter_cv25_split(lang_dir: Path, clip_names: set[str], clip_durations: dict[str, float]) -> Iterator[tuple[Path, float, str]]:
    """Yield (mp3_path, duration_seconds, sentence_text) for clips in `clip_names`."""
    # Read whichever TSV contains sentences — the union of train/dev/test/validated/other
    # is where the sentence_text lives.
    sentence_by_clip: dict[str, str] = {}
    for tsv in ["train.tsv", "dev.tsv", "test.tsv", "validated.tsv", "other.tsv"]:
        p = lang_dir / tsv
        if not p.exists():
            continue
        with p.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                c = (row.get("path") or "").strip()
                if not c or c in sentence_by_clip:
                    continue
                s = (row.get("sentence") or row.get("sentence_raw") or "").strip()
                if s:
                    sentence_by_clip[c] = s

    clips_dir = lang_dir / "clips"
    missing = 0
    for c in sorted(clip_names):
        mp3 = clips_dir / c
        if not mp3.exists():
            missing += 1
            continue
        sent = sentence_by_clip.get(c, "")
        if not sent:
            missing += 1
            continue
        dur = clip_durations.get(c, 0.0)
        yield mp3, dur, sent
    if missing:
        print(f"  (skipped {missing} clips missing audio or text)", flush=True)


def cv25_clip_set(lang_dir: Path, tsv_name: str) -> set[str]:
    out: set[str] = set()
    p = lang_dir / tsv_name
    if not p.exists():
        return out
    with p.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            c = (row.get("path") or "").strip()
            if c:
                out.add(c)
    return out


def cv25_clip_durations(lang_dir: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with (lang_dir / "clip_durations.tsv").open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                out[row[0].strip()] = float(row[1]) / 1000.0
            except ValueError:
                pass
    return out


def prepare_cv25_lt(
    cv25_dir: Path, out_dir: Path, audio_dir: Path, gcs_uri: str, max_clips: int | None = None
) -> dict[str, Path]:
    """Build CV25 LT train_bundle / dev / test manifests. Returns {split: manifest_path}."""
    lang_dir = ensure_cv25(cv25_dir, gcs_uri)
    clip_durations = cv25_clip_durations(lang_dir)

    train_bundle = (
        cv25_clip_set(lang_dir, "train.tsv")
        | (cv25_clip_set(lang_dir, "validated.tsv") - cv25_clip_set(lang_dir, "dev.tsv") - cv25_clip_set(lang_dir, "test.tsv"))
        | cv25_clip_set(lang_dir, "other.tsv")
    )
    train_bundle -= cv25_clip_set(lang_dir, "test.tsv")
    train_bundle -= cv25_clip_set(lang_dir, "dev.tsv")

    splits: dict[str, set[str]] = {
        "train": train_bundle,
        "dev": cv25_clip_set(lang_dir, "dev.tsv"),
        "test": cv25_clip_set(lang_dir, "test.tsv"),
    }

    audio_out = audio_dir / "cv25_lt"
    audio_out.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for split_name, clips in splits.items():
        # For dev/test we copy mp3s as-is (no decode cost in manifests —
        # NeMo reads the file at training time). For train we still copy
        # the raw mp3 to the per-dataset audio dir so everything is in
        # one place, but we don't actually re-encode.
        manifest_path = out_dir / f"cv25_lt_{split_name}.json"
        print(f"▸ CV25 LT {split_name}: {len(clips):,} clips → {manifest_path.name}", flush=True)
        n = 0
        with manifest_path.open("w", encoding="utf-8") as fout:
            for mp3, dur, sent in iter_cv25_split(lang_dir, clips, clip_durations):
                if max_clips and n >= max_clips:
                    break
                # Use the original MP3 path — NeMo can decode MP3 directly
                # and it saves us from re-encoding ~18 GB of audio.
                text = clean_text(sent)
                if not text:
                    continue
                # Keep the manifest path absolute so it works from any cwd.
                fout.write(
                    json.dumps(
                        {
                            "audio_filepath": str(mp3.resolve()),
                            "duration": dur,
                            "text": text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n += 1
        print(f"  wrote {n:,} entries", flush=True)
        written[split_name] = manifest_path
    return written


# ─── HF-backed loaders ─────────────────────────────────────────────────────

def prepare_hf_dataset(
    hf_id: str,
    config: str | None,
    splits: list[str],
    audio_field: str,
    text_field: str,
    out_prefix: str,
    out_dir: Path,
    audio_dir: Path,
    max_clips: int | None = None,
    language_filter: str | None = None,
) -> dict[str, Path]:
    """Generic HF dataset → manifest.

    Streams the split (avoids materializing the whole thing), writes each
    clip as a 16 kHz WAV under `audio_dir/<out_prefix>/`, and emits
    a manifest line per clip.
    """
    from datasets import load_dataset  # type: ignore

    written: dict[str, Path] = {}
    audio_out = audio_dir / out_prefix
    audio_out.mkdir(parents=True, exist_ok=True)

    for split in splits:
        manifest_path = out_dir / f"{out_prefix}_{split}.json"
        print(f"▸ {hf_id} [{config or '-'}] split={split} → {manifest_path.name}", flush=True)
        # VoxPopuli / FLEURS ship loading scripts and require
        # trust_remote_code=True on newer `datasets` versions.
        kwargs: dict = {"streaming": True, "trust_remote_code": True}
        try:
            if config:
                ds = load_dataset(hf_id, config, split=split, **kwargs)
            else:
                ds = load_dataset(hf_id, split=split, **kwargs)
        except Exception as e:
            print(f"  SKIP ({e.__class__.__name__}): {e}", flush=True)
            continue

        n = 0
        with manifest_path.open("w", encoding="utf-8") as fout:
            for i, sample in enumerate(ds):
                if max_clips and n >= max_clips:
                    break
                # Optional language filter (voxpopuli ships other langs too
                # if pulled via the wrong config — we set config=lt so
                # this is a no-op in practice).
                if language_filter:
                    lang = sample.get("language") or sample.get("lang") or ""
                    if str(lang).lower() != language_filter:
                        continue
                audio = sample.get(audio_field)
                if not audio:
                    continue
                if isinstance(audio, dict):
                    arr = audio.get("array")
                    sr = audio.get("sampling_rate") or 16000
                else:
                    arr = audio
                    sr = 16000
                if arr is None:
                    continue

                text = sample.get(text_field)
                if not text:
                    continue
                text = clean_text(str(text))
                if not text:
                    continue

                clip_id = (
                    sample.get("audio_id")
                    or sample.get("id")
                    or sample.get("utt_id")
                    or sample.get("__key__")
                    or f"{out_prefix}_{split}_{i:07d}"
                )
                wav_path = audio_out / f"{clip_id}.wav"
                try:
                    dur = write_wav_mono16k(arr, int(sr), wav_path)
                except Exception as e:
                    print(f"  skip {clip_id}: {e}", flush=True)
                    continue
                fout.write(
                    json.dumps(
                        {
                            "audio_filepath": str(wav_path.resolve()),
                            "duration": dur,
                            "text": text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n += 1
                if n % 500 == 0:
                    print(f"  {n:,}…", flush=True)

        print(f"  wrote {n:,} entries", flush=True)
        written[split] = manifest_path
    return written


# ─── Concatenation helper ──────────────────────────────────────────────────


def concat_and_shuffle(inputs: list[Path], out_path: Path, seed: int = 1234) -> None:
    lines: list[str] = []
    for p in inputs:
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            lines.extend(x for x in f if x.strip())
    random.Random(seed).shuffle(lines)
    with out_path.open("w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"▸ concat {len(lines):,} lines → {out_path}", flush=True)


# ─── Main ──────────────────────────────────────────────────────────────────

KNOWN = {"cv25_lt", "voxpopuli_lt", "fleurs_lt", "shunyalabs_lt"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cv25-dir", type=Path, default=DEFAULT_CV25_DIR)
    ap.add_argument("--cv25-gcs", default=DEFAULT_CV25_GCS)
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["cv25_lt"],
        help=f"Subset of: {sorted(KNOWN)}",
    )
    ap.add_argument("--max-clips", type=int, default=None, help="Per-split clip cap (smoke tests)")
    args = ap.parse_args()

    unknown = set(args.datasets) - KNOWN
    if unknown:
        print(f"error: unknown datasets: {unknown}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    audio_dir = args.out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    all_train: list[Path] = []
    all_test: list[Path] = []

    if "cv25_lt" in args.datasets:
        m = prepare_cv25_lt(args.cv25_dir, args.out, audio_dir, args.cv25_gcs, args.max_clips)
        if "train" in m:
            all_train.append(m["train"])
        if "test" in m:
            all_test.append(m["test"])

    if "voxpopuli_lt" in args.datasets:
        m = prepare_hf_dataset(
            hf_id="facebook/voxpopuli",
            config="lt",
            splits=["train", "test"],
            audio_field="audio",
            text_field="normalized_text",
            out_prefix="voxpopuli_lt",
            out_dir=args.out,
            audio_dir=audio_dir,
            max_clips=args.max_clips,
        )
        if "train" in m:
            all_train.append(m["train"])
        if "test" in m:
            all_test.append(m["test"])

    if "fleurs_lt" in args.datasets:
        m = prepare_hf_dataset(
            hf_id="google/fleurs",
            config="lt_lt",
            splits=["train", "test"],
            audio_field="audio",
            text_field="transcription",
            out_prefix="fleurs_lt",
            out_dir=args.out,
            audio_dir=audio_dir,
            max_clips=args.max_clips,
        )
        if "train" in m:
            all_train.append(m["train"])
        if "test" in m:
            all_test.append(m["test"])

    if "shunyalabs_lt" in args.datasets:
        m = prepare_hf_dataset(
            hf_id="shunyalabs/lithuanian-speech-dataset",
            config=None,
            splits=["train", "test"],
            audio_field="audio",
            text_field="transcript",
            out_prefix="shunyalabs_lt",
            out_dir=args.out,
            audio_dir=audio_dir,
            max_clips=args.max_clips,
        )
        if "train" in m:
            all_train.append(m["train"])
        if "test" in m:
            all_test.append(m["test"])

    if all_train:
        concat_and_shuffle(all_train, args.out / "ALL_train.json")
    if all_test:
        concat_and_shuffle(all_test, args.out / "ALL_test.json")

    print("▸ done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
