#!/usr/bin/env python3
"""Evaluate a NeMo ASR model using the Open ASR Leaderboard methodology.

Uses the EXACT normalizer and datasets that the HuggingFace Open ASR
Leaderboard uses, so numbers are directly comparable to published results
from Whisper, Parakeet, Canary, etc.

Key difference from 04_eval.py:
  - Uses `whisper_normalizer.BasicTextNormalizer` (the leaderboard standard)
    instead of our custom punctuation-stripping normalizer.
  - Loads datasets directly from HuggingFace (not local manifests) to
    ensure the exact same test split and audio processing.
  - Reports in the same format the leaderboard expects.

Install:
    pip install whisper-normalizer datasets jiwer nemo_toolkit[asr] soundfile

Run:
    # Eval on FLEURS Lithuanian (the primary multilingual benchmark)
    python scripts/12_eval_leaderboard.py \\
        --model sliderforthewin/parakeet-tdt-lt \\
        --dataset google/fleurs --config lt_lt --split test \\
        --out results/leaderboard_fleurs_lt

    # Eval on Common Voice (specify version)
    python scripts/12_eval_leaderboard.py \\
        --model sliderforthewin/parakeet-tdt-lt \\
        --dataset mozilla-foundation/common_voice_17_0 --config lt --split test \\
        --out results/leaderboard_cv17_lt

    # Eval on VoxPopuli
    python scripts/12_eval_leaderboard.py \\
        --model sliderforthewin/parakeet-tdt-lt \\
        --dataset facebook/voxpopuli --config lt --split test \\
        --text-field normalized_text \\
        --out results/leaderboard_voxpopuli_lt

    # Also eval the stock model for comparison
    python scripts/12_eval_leaderboard.py \\
        --model nvidia/parakeet-tdt-0.6b-v3 \\
        --dataset google/fleurs --config lt_lt --split test \\
        --out results/leaderboard_fleurs_lt_baseline
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model ID or local .nemo path")
    ap.add_argument("--dataset", required=True, help="HF dataset ID (e.g. google/fleurs)")
    ap.add_argument("--config", default=None, help="HF dataset config (e.g. lt_lt)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--text-field", default="transcription",
                    help="Dataset column with reference text (transcription for FLEURS, "
                         "sentence for CV, normalized_text for VoxPopuli)")
    ap.add_argument("--audio-field", default="audio")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lm", type=Path, default=None,
                    help="Path to KenLM ARPA file for beam+LM fusion")
    ap.add_argument("--beam-size", type=int, default=None,
                    help="Beam size (enables beam search; requires --lm for LM fusion)")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="LM weight for beam+LM fusion")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # ── Load the leaderboard normalizer ──────────────────────────────
    from whisper_normalizer.basic import BasicTextNormalizer
    normalizer = BasicTextNormalizer()
    print(f"▸ normalizer: whisper BasicTextNormalizer (Open ASR Leaderboard standard)", flush=True)

    # ── Load dataset from HuggingFace ────────────────────────────────
    from datasets import load_dataset
    print(f"▸ loading {args.dataset} [{args.config}] split={args.split}", flush=True)

    kwargs = {"trust_remote_code": True}
    if args.config:
        ds = load_dataset(args.dataset, args.config, split=args.split, **kwargs)
    else:
        ds = load_dataset(args.dataset, split=args.split, **kwargs)

    if args.max_clips:
        ds = ds.select(range(min(args.max_clips, len(ds))))
    print(f"  {len(ds):,} clips", flush=True)

    # ── Decode audio to temp WAV files (NeMo needs file paths) ───────
    import torch
    print(f"▸ decoding audio to temp WAVs...", flush=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="leaderboard_eval_"))
    file_paths = []
    refs_raw = []
    durations = []

    for i, sample in enumerate(ds):
        audio = sample[args.audio_field]
        if isinstance(audio, dict):
            arr = np.asarray(audio["array"], dtype="float32")
            sr = audio["sampling_rate"]
        else:
            arr = np.asarray(audio, dtype="float32")
            sr = 16000

        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        if sr != 16000:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)

        wav_path = tmp_dir / f"{i:06d}.wav"
        sf.write(str(wav_path), arr, 16000, subtype="PCM_16")
        file_paths.append(str(wav_path))
        durations.append(len(arr) / 16000.0)

        ref = sample.get(args.text_field) or ""
        refs_raw.append(str(ref))

        if (i + 1) % 500 == 0:
            print(f"  {i+1:,}...", flush=True)

    total_audio = sum(durations)
    print(f"  {len(file_paths):,} clips, {total_audio:.0f}s audio", flush=True)

    # ── Load model ───────────────────────────────────────────────────
    import nemo.collections.asr as nemo_asr
    model_arg = args.model
    if model_arg.endswith(".nemo") and Path(model_arg).exists():
        print(f"▸ loading local: {model_arg}", flush=True)
        model = nemo_asr.models.ASRModel.restore_from(model_arg)
    else:
        print(f"▸ loading from HF: {model_arg}", flush=True)
        model = nemo_asr.models.ASRModel.from_pretrained(model_arg)

    if torch.cuda.is_available():
        model = model.to("cuda")
        if torch.cuda.is_bf16_supported():
            model = model.to(torch.bfloat16)
    model.eval()

    # ── Optional: switch to beam+LM decoding ─────────────────────────
    decoding_label = "greedy"
    if args.beam_size:
        import copy
        from omegaconf import open_dict
        new_cfg = copy.deepcopy(model.cfg.decoding)
        with open_dict(new_cfg):
            new_cfg.strategy = "maes"
            new_cfg.beam.beam_size = args.beam_size
            new_cfg.beam.return_best_hypothesis = True
            if args.lm and args.lm.exists():
                new_cfg.beam.ngram_lm_model = str(args.lm)
                new_cfg.beam.ngram_lm_alpha = args.alpha
                decoding_label = f"beam={args.beam_size}+LM(α={args.alpha})"
            else:
                decoding_label = f"beam={args.beam_size}"
        model.change_decoding_strategy(new_cfg)
        print(f"▸ decoding: {decoding_label}", flush=True)

    # ── Transcribe ───────────────────────────────────────────────────
    print(f"▸ transcribing (batch_size={args.batch_size})...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        outs = model.transcribe(file_paths, batch_size=args.batch_size, verbose=False)
    elapsed = time.time() - t0
    rtfx = total_audio / elapsed if elapsed > 0 else 0

    hyps_raw = []
    for item in outs:
        if isinstance(item, list):
            item = item[0] if item else ""
        if hasattr(item, "text"):
            item = item.text
        if isinstance(item, tuple):
            item = item[0]
        hyps_raw.append(str(item))

    print(f"  {elapsed:.1f}s ({rtfx:.1f}× real-time)", flush=True)

    # ── Normalize with the leaderboard normalizer ────────────────────
    refs_norm = [normalizer(r).strip() for r in refs_raw]
    hyps_norm = [normalizer(h).strip() for h in hyps_raw]

    # ── Compute WER/CER ──────────────────────────────────────────────
    import jiwer
    pairs = [(r, h) for r, h in zip(refs_norm, hyps_norm) if r.strip()]
    if not pairs:
        print("error: no non-empty references after normalization", file=sys.stderr)
        return 2
    r_list = [p[0] for p in pairs]
    h_list = [p[1] for p in pairs]
    wer = jiwer.wer(r_list, h_list)
    cer = jiwer.cer(r_list, h_list)

    # ── Report ───────────────────────────────────────────────────────
    print("─" * 60)
    print(f"  Open ASR Leaderboard-compatible eval")
    print(f"  model:      {args.model}")
    print(f"  dataset:    {args.dataset} [{args.config}] {args.split}")
    print(f"  decoding:   {decoding_label}")
    print(f"  normalizer: whisper BasicTextNormalizer")
    print(f"  n_clips:    {len(pairs):,}")
    print(f"  WER:        {wer*100:.2f}%")
    print(f"  CER:        {cer*100:.2f}%")
    print(f"  RTFx:       {rtfx:.1f}×")
    print("─" * 60)

    # ── Save results ─────────────────────────────────────────────────
    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "decoding": decoding_label,
        "normalizer": "whisper_normalizer.BasicTextNormalizer",
        "n_clips": len(pairs),
        "audio_seconds": total_audio,
        "wall_seconds": elapsed,
        "rtfx": rtfx,
        "wer": wer,
        "cer": cer,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    with (args.out / "per_clip.jsonl").open("w", encoding="utf-8") as f:
        for ref, hyp, ref_n, hyp_n, dur in zip(refs_raw, hyps_raw, refs_norm, hyps_norm, durations):
            f.write(json.dumps({
                "reference_raw": ref,
                "hypothesis_raw": hyp,
                "reference_norm": ref_n,
                "hypothesis_norm": hyp_n,
                "duration": dur,
            }, ensure_ascii=False) + "\n")

    # ── Cleanup temp files ───────────────────────────────────────────
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"▸ results → {args.out}/", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
