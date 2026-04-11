#!/usr/bin/env python3
"""Eval parakeet with tokenizer masking to force Lithuanian-only output.

Intercepts the model's joint-network logits and sets non-Lithuanian
tokens to -inf before the argmax. This prevents catastrophic drift
into Cyrillic/Romanian/etc that accounts for ~149 clips (2.6%) of
the CV25 LT test set with >100% WER.

Run:
    python scripts/09_eval_masked.py \\
        --model nvidia/parakeet-tdt-0.6b-v3 \\
        --manifest data/manifests/cv25_lt_test.json \\
        --allowed-tokens data/lm/lt_allowed_tokens.pkl \\
        --out results/baseline_masked
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
import unicodedata
from pathlib import Path

import torch

PUNCT_RE = re.compile(r"[\"\'\,\.\!\?\;\:\(\)\[\]\{\}\«\»\"\"\„\‟\‚\'\'\–\—\-]")
SPACE_RE = re.compile(r"\s+")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    s = PUNCT_RE.sub(" ", s)
    return SPACE_RE.sub(" ", s).strip()


def install_logit_mask(model, allowed: set[int]) -> None:
    """Monkey-patch joint.joint_after_projection to mask non-LT tokens.

    During greedy decoding, the TDT decoder calls joint.joint_after_projection
    which returns a tensor of shape [..., vocab_size + 1] — labels + blank.
    Durations are computed separately and aren't in this tensor.

    We mask label positions that aren't in the allowed LT set, leaving blank
    unmasked so the decoder can still terminate tokens.
    """
    joint = model.joint
    vocab_size = joint.num_classes_with_blank - 1  # real labels (excludes blank)
    total = vocab_size + 1  # labels + blank

    blocked = torch.zeros(total, dtype=torch.bool)
    for i in range(vocab_size):
        if i not in allowed:
            blocked[i] = True
    # blank at index vocab_size stays unmasked
    n_blocked = int(blocked.sum().item())
    print(f"  joint: vocab={vocab_size} blank_idx={vocab_size}", flush=True)
    print(f"  masking {n_blocked}/{vocab_size} label tokens", flush=True)

    mask_value = -1e9

    # Patch both entry points:
    # - joint_after_projection: used by greedy/TDT batched decoders (post-proj)
    # - joint: used by non-batched decoders (includes projection internally)
    orig_jap = joint.joint_after_projection
    def masked_jap(f, g):
        out = orig_jap(f, g)
        if isinstance(out, torch.Tensor) and out.shape[-1] == total:
            out = out.masked_fill(blocked.to(out.device), mask_value)
        return out
    joint.joint_after_projection = masked_jap

    orig_j = joint.joint
    def masked_j(f, g):
        out = orig_j(f, g)
        if isinstance(out, torch.Tensor) and out.shape[-1] == total:
            out = out.masked_fill(blocked.to(out.device), mask_value)
        return out
    joint.joint = masked_j


def transcribe(model, files: list[str], batch_size: int = 8) -> list[str]:
    model.eval()
    with torch.no_grad():
        outs = model.transcribe(files, batch_size=batch_size, verbose=False)
    hyps = []
    for item in outs:
        if isinstance(item, list):
            item = item[0] if item else ""
        if hasattr(item, "text"):
            item = item.text
        if isinstance(item, tuple):
            item = item[0]
        hyps.append(str(item))
    return hyps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--allowed-tokens", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-clips", type=int, default=None)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"▸ loading allowed tokens from {args.allowed_tokens}", flush=True)
    with args.allowed_tokens.open("rb") as f:
        allowed = pickle.load(f)
    print(f"  {len(allowed)} allowed tokens", flush=True)

    print(f"▸ loading model {args.model}", flush=True)
    import nemo.collections.asr as nemo_asr
    if args.model.endswith(".nemo") and Path(args.model).exists():
        model = nemo_asr.models.ASRModel.restore_from(args.model)
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(args.model)
    model = model.to("cuda")

    print(f"▸ installing logit mask", flush=True)
    install_logit_mask(model, allowed)

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

    print(f"▸ transcribing...", flush=True)
    t0 = time.time()
    hyps_raw = transcribe(model, files, batch_size=args.batch_size)
    elapsed = time.time() - t0
    total_audio = sum(c["duration"] for c in clips)
    print(f"  {elapsed:.1f}s  ({total_audio/elapsed:.1f}× real-time)", flush=True)

    # Compute WER
    import jiwer
    refs = [_norm(r) for r in refs_raw]
    hyps = [_norm(h) for h in hyps_raw]
    wer = jiwer.wer(refs, hyps) * 100
    cer = jiwer.cer(refs, hyps) * 100

    print("─" * 60)
    print(f"  model:      {args.model}")
    print(f"  masked:     yes ({len(allowed)} tokens)")
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
            "model": args.model,
            "manifest": str(args.manifest),
            "n_clips": len(clips),
            "WER": wer,
            "CER": cer,
            "masked": True,
            "allowed_tokens": len(allowed),
        }, f, indent=2)
    print(f"▸ saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
