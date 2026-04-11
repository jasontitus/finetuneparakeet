#!/usr/bin/env python3
"""Evaluate a NeMo ASR model on a Lithuanian manifest.

Works for both the stock `nvidia/parakeet-tdt-0.6b-v3` baseline and
any fine-tuned `.nemo` checkpoint we produce. Outputs per-clip JSON
with `reference_raw`, `hypothesis_raw`, `reference_norm`, and
`hypothesis_norm`, plus an aggregate `summary.json` with WER/CER.

Run (VM-side):

    python scripts/04_eval.py \
        --model nvidia/parakeet-tdt-0.6b-v3 \
        --manifest data/manifests/cv25_lt_test.json \
        --out results/baseline_cv25_lt_test \
        [--max-clips 200] [--batch-size 8]

The script is careful to not hold every decoded audio in RAM — it
batches through `NeMoASRModel.transcribe(batch_size=...)` which
accepts a list of file paths and streams internally.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

# Normalization must match prepare_manifests.py exactly so that a
# baseline and a fine-tuned model can be compared apples-to-apples.
PUNCT_RE = re.compile(r"[\"\'\,\.\!\?\;\:\(\)\[\]\{\}\«\»\“\”\„\‟\‚\‘\’\–\—\-]")
SPACE_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = s.lower()
    s = PUNCT_RE.sub(" ", s)
    s = SPACE_RE.sub(" ", s).strip()
    return s


def load_manifest(path: Path, max_clips: int | None) -> list[dict]:
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.append(rec)
            if max_clips and len(out) >= max_clips:
                break
    return out


def load_model(model_arg: str):
    """Load a NeMo ASR model, either from a HF id or a local .nemo file."""
    import nemo.collections.asr as nemo_asr  # type: ignore
    import torch  # type: ignore

    if model_arg.endswith(".nemo") and Path(model_arg).exists():
        print(f"▸ loading from local checkpoint {model_arg}", flush=True)
        model = nemo_asr.models.ASRModel.restore_from(model_arg)
    else:
        print(f"▸ loading pretrained {model_arg}", flush=True)
        model = nemo_asr.models.ASRModel.from_pretrained(model_arg)

    if torch.cuda.is_available():
        model = model.to("cuda")
        try:
            model = model.to(torch.bfloat16) if torch.cuda.is_bf16_supported() else model.to(torch.float16)
        except Exception:
            pass
    model.eval()
    return model


def transcribe_batch(model, file_paths: list[str], batch_size: int) -> list[str]:
    """Run NeMo transcribe on a list of file paths and normalize outputs to strings."""
    outs = model.transcribe(file_paths, batch_size=batch_size, verbose=False)
    result: list[str] = []
    for item in outs:
        # NeMo 2.x returns a Hypothesis object; 1.x returns str or list.
        if isinstance(item, list):
            item = item[0] if item else ""
        if hasattr(item, "text"):
            item = item.text
        if isinstance(item, tuple):
            item = item[0]
        result.append(str(item))
    return result


def compute_wer_cer(refs: list[str], hyps: list[str]) -> tuple[float, float]:
    import jiwer  # type: ignore

    # Guard against empty references — jiwer errors on those.
    pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    if not pairs:
        return float("nan"), float("nan")
    r = [p[0] for p in pairs]
    h = [p[1] for p in pairs]
    wer = jiwer.wer(r, h)
    cer = jiwer.cer(r, h)
    return wer, cer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or path to a local .nemo file")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="Output directory for per-clip + summary JSON")
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    records = load_manifest(args.manifest, args.max_clips)
    if not records:
        print("error: no records in manifest", file=sys.stderr)
        return 2
    print(f"▸ {len(records):,} clips from {args.manifest}", flush=True)

    model = load_model(args.model)
    print("▸ model loaded", flush=True)

    t0 = time.time()
    file_paths = [r["audio_filepath"] for r in records]
    hyps_raw = transcribe_batch(model, file_paths, args.batch_size)
    elapsed = time.time() - t0
    audio_s = sum(float(r.get("duration", 0.0)) for r in records)
    rtfx = audio_s / elapsed if elapsed > 0 else float("nan")
    print(f"▸ transcribed {len(records):,} clips in {elapsed:.1f}s ({rtfx:.1f}× real-time)", flush=True)

    # Write per-clip file + compute aggregate
    per_clip_path = args.out / "per_clip.jsonl"
    refs_norm: list[str] = []
    hyps_norm: list[str] = []
    with per_clip_path.open("w", encoding="utf-8") as f:
        for rec, hyp_raw in zip(records, hyps_raw):
            ref_raw = rec.get("text", "")
            ref_n = normalize_text(ref_raw)
            hyp_n = normalize_text(hyp_raw)
            refs_norm.append(ref_n)
            hyps_norm.append(hyp_n)
            f.write(
                json.dumps(
                    {
                        "audio_filepath": rec.get("audio_filepath"),
                        "duration": rec.get("duration"),
                        "reference_raw": ref_raw,
                        "reference_norm": ref_n,
                        "hypothesis_raw": hyp_raw,
                        "hypothesis_norm": hyp_n,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    wer, cer = compute_wer_cer(refs_norm, hyps_norm)
    summary = {
        "model": args.model,
        "manifest": str(args.manifest),
        "n_clips": len(records),
        "audio_seconds": audio_s,
        "wall_seconds": elapsed,
        "rtfx": rtfx,
        "wer": wer,
        "cer": cer,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("─" * 60)
    print(f"  model:   {args.model}")
    print(f"  n_clips: {len(records):,}")
    print(f"  WER:     {wer:.4f}")
    print(f"  CER:     {cer:.4f}")
    print(f"  RTFx:    {rtfx:.1f}×")
    print("─" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
