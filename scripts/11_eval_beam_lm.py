#!/usr/bin/env python3
"""Evaluate a NeMo TDT ASR model with beam search + n-gram LM fusion.

Switches decoding strategy from greedy_batch to maes (Modified Adaptive
Expansion Search), which is the only TDT strategy that supports
ngram_lm_model fusion in NeMo. Scores each beam candidate as:

    combined = acoustic_score + alpha * LM_score

Run:
    python scripts/11_eval_beam_lm.py \\
        --model nvidia/parakeet-tdt-0.6b-v3 \\
        --manifest data/manifests/cv25_lt_dev.json \\
        --lm data/lm/lt_wiki_4gram.arpa \\
        --beam-size 8 \\
        --alpha 0.5 \\
        --out results/baseline_beamlm_dev \\
        [--max-clips 200]

Hyperparameter sweep (tune alpha on a small subset):
    for alpha in 0.0 0.2 0.3 0.5 0.7 1.0; do
        python scripts/11_eval_beam_lm.py \\
            --model <...> --manifest <...> --lm <...> \\
            --beam-size 8 --alpha $alpha --max-clips 200 \\
            --out results/sweep_alpha_$alpha
    done
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import torch
from omegaconf import OmegaConf, open_dict

PUNCT_RE = re.compile(r"[\"\'\,\.\!\?\;\:\(\)\[\]\{\}\«\»\"\"\„\‟\‚\'\'\–\—\-]")
SPACE_RE = re.compile(r"\s+")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    s = PUNCT_RE.sub(" ", s)
    return SPACE_RE.sub(" ", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--lm", type=Path, required=True,
                    help="Path to KenLM ARPA file")
    ap.add_argument("--beam-size", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="LM weight (0 = no LM, 0.3 default, 0.5-1.0 aggressive)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-clips", type=int, default=None)
    args = ap.parse_args()

    assert args.lm.exists(), f"LM file not found: {args.lm}"
    args.out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"▸ device: {device}", flush=True)
    print(f"▸ loading model: {args.model}", flush=True)
    import nemo.collections.asr as nemo_asr
    if str(args.model).endswith(".nemo") and Path(args.model).exists():
        model = nemo_asr.models.ASRModel.restore_from(
            str(args.model), map_location=device
        )
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(
            args.model, map_location=device
        )
    model = model.to(device)
    model.eval()

    # Switch decoding strategy to maes (TDT beam search with LM fusion).
    print(f"▸ switching to beam search (maes) + LM", flush=True)
    new_cfg = copy.deepcopy(model.cfg.decoding)
    with open_dict(new_cfg):
        new_cfg.strategy = "maes"
        new_cfg.beam.beam_size = args.beam_size
        new_cfg.beam.return_best_hypothesis = True
        new_cfg.beam.ngram_lm_model = str(args.lm)
        new_cfg.beam.ngram_lm_alpha = args.alpha
    print(f"  strategy: {new_cfg.strategy}", flush=True)
    print(f"  beam_size: {new_cfg.beam.beam_size}", flush=True)
    print(f"  alpha: {new_cfg.beam.ngram_lm_alpha}", flush=True)
    model.change_decoding_strategy(new_cfg)

    clips = []
    with args.manifest.open() as f:
        for line in f:
            if line.strip():
                clips.append(json.loads(line))
                if args.max_clips and len(clips) >= args.max_clips:
                    break
    print(f"▸ {len(clips):,} clips", flush=True)

    files = [c["audio_filepath"] for c in clips]
    refs_raw = [c["text"] for c in clips]

    print(f"▸ transcribing (beam + LM)...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        outs = model.transcribe(files, batch_size=args.batch_size, verbose=False)
    elapsed = time.time() - t0
    total_audio = sum(c["duration"] for c in clips)
    print(f"  {elapsed:.1f}s  ({total_audio/elapsed:.1f}× real-time)", flush=True)

    hyps_raw = []
    for item in outs:
        if isinstance(item, list):
            item = item[0] if item else ""
        if hasattr(item, "text"):
            item = item.text
        if isinstance(item, tuple):
            item = item[0]
        hyps_raw.append(str(item))

    import jiwer
    refs = [_norm(r) for r in refs_raw]
    hyps = [_norm(h) for h in hyps_raw]
    wer = jiwer.wer(refs, hyps) * 100
    cer = jiwer.cer(refs, hyps) * 100

    print("─" * 60)
    print(f"  model:      {args.model}")
    print(f"  lm:         {args.lm.name}")
    print(f"  beam_size:  {args.beam_size}")
    print(f"  alpha:      {args.alpha}")
    print(f"  n_clips:    {len(clips):,}")
    print(f"  WER:        {wer:.2f}%")
    print(f"  CER:        {cer:.2f}%")
    print("─" * 60)

    # Save per-clip for error analysis
    with (args.out / "per_clip.jsonl").open("w") as f:
        for c, ref, hyp in zip(clips, refs_raw, hyps_raw):
            f.write(json.dumps({
                "audio_filepath": c["audio_filepath"],
                "duration": c["duration"],
                "reference_raw": ref,
                "reference_norm": _norm(ref),
                "hypothesis_raw": hyp,
                "hypothesis_norm": _norm(hyp),
            }, ensure_ascii=False) + "\n")

    with (args.out / "summary.json").open("w") as f:
        json.dump({
            "model": str(args.model),
            "manifest": str(args.manifest),
            "lm": str(args.lm),
            "beam_size": args.beam_size,
            "alpha": args.alpha,
            "n_clips": len(clips),
            "WER": wer,
            "CER": cer,
        }, f, indent=2)
    print(f"▸ saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
