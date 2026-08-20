#!/usr/bin/env python3
"""Transcribe audio with the fine-tuned Lithuanian parakeet-tdt model.

Simple standalone runner for the model published at
`sliderforthewin/parakeet-tdt-lt` on HuggingFace Hub. Handles downloading,
device placement, optional LM fusion, long-audio chunking, and printing results.

Recordings longer than 90s are split on detected silence before decoding. The
model was fine-tuned and evaluated on Common Voice clips of a few seconds; fed a
whole hour-long file it silently drops large spans of speech (measured: 168 words
returned for 7m41s of near-continuous speech, vs 793 words for the same audio
chunked). Pass --no-chunk to reproduce the old whole-file behaviour.

Examples:
    # Single file, greedy decoding (14.06% WER on CV25 LT test)
    python scripts/transcribe.py audio.wav

    # Multiple files
    python scripts/transcribe.py clip1.wav clip2.wav clip3.wav

    # Full accuracy — beam search + token-level LM (11.23% WER)
    python scripts/transcribe.py --lm audio.wav

    # Use a local .nemo checkpoint instead of downloading
    python scripts/transcribe.py --model /path/to/local.nemo audio.wav

    # Use a local LM file instead of downloading
    python scripts/transcribe.py --lm --lm-path /path/to/lm.arpa audio.wav

    # Force CPU (slower, useful for debugging)
    python scripts/transcribe.py --device cpu audio.wav

    # JSON output (adds per-segment start/end when chunking kicks in)
    python scripts/transcribe.py --json audio.wav

    # Tune or disable chunking
    python scripts/transcribe.py --chunk-seconds 30 long_interview.mp3
    python scripts/transcribe.py --no-chunk short_clip.wav
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path


DEFAULT_MODEL = "sliderforthewin/parakeet-tdt-lt"
DEFAULT_LM_FILENAME = "lt_token_4gram.arpa"

# Long-audio chunking. The model was fine-tuned and evaluated on Common Voice
# clips (a few seconds each); handing it a whole hour-long recording makes it
# silently drop large spans, so anything longer than CHUNK_THRESHOLD_S is split.
# Cuts land inside detected silence so words are not sliced in half.
CHUNK_THRESHOLD_S = 90.0
CHUNK_TARGET_S = 45.0
CHUNK_MAX_S = 75.0
CHUNK_MIN_S = 5.0
SILENCE_DB = -35
SILENCE_MIN_S = 0.30


def load_model(model_arg: str, device: str):
    """Load a NeMo ASR model from HF Hub, local .nemo file, or HF model id."""
    import nemo.collections.asr as nemo_asr

    if model_arg.endswith(".nemo") and Path(model_arg).exists():
        print(f"▸ loading local checkpoint: {model_arg}", file=sys.stderr)
        model = nemo_asr.models.ASRModel.restore_from(model_arg, map_location=device)
    else:
        print(f"▸ loading {model_arg} from HuggingFace...", file=sys.stderr)
        model = nemo_asr.models.ASRModel.from_pretrained(model_arg, map_location=device)

    model = model.to(device)
    model.eval()
    return model


def enable_lm(model, lm_path: str, beam_size: int = 4, alpha: float = 0.3):
    """Switch decoder to maes beam search + n-gram LM fusion."""
    from omegaconf import open_dict

    cfg = copy.deepcopy(model.cfg.decoding)
    with open_dict(cfg):
        cfg.strategy = "maes"
        cfg.beam.beam_size = beam_size
        cfg.beam.return_best_hypothesis = True
        cfg.beam.ngram_lm_model = lm_path
        cfg.beam.ngram_lm_alpha = alpha
    model.change_decoding_strategy(cfg)
    print(
        f"▸ beam+LM enabled (beam_size={beam_size}, alpha={alpha})",
        file=sys.stderr,
    )


def resolve_lm_path(arg_path: str | None, model_arg: str) -> str:
    """Return a local path to the LM file, downloading from HF if needed."""
    if arg_path:
        p = Path(arg_path)
        if not p.exists():
            raise FileNotFoundError(f"LM file not found: {p}")
        return str(p)

    # Try to download from the same HF repo as the model
    if model_arg.endswith(".nemo"):
        raise ValueError(
            "Pass --lm-path explicitly when using a local .nemo model"
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "Need `huggingface_hub` installed to download the LM file.\n"
            "Run: pip install huggingface_hub"
        )
    print(f"▸ downloading LM from {model_arg}/{DEFAULT_LM_FILENAME}...", file=sys.stderr)
    return hf_hub_download(repo_id=model_arg, filename=DEFAULT_LM_FILENAME)


def probe_duration(path: str) -> float | None:
    """Return duration in seconds via ffprobe, or None if it cannot be read."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return float(out)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def detect_silences(path: str) -> list[tuple[float, float]]:
    """Return [(start, end)] of silent stretches, via ffmpeg silencedetect."""
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-v", "info", "-i", path,
        "-af", f"silencedetect=n={SILENCE_DB}dB:d={SILENCE_MIN_S}",
        "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError:
        return []
    spans: list[tuple[float, float]] = []
    start: float | None = None
    for line in proc.stderr.splitlines():
        if "silence_start:" in line:
            try:
                start = float(line.rsplit("silence_start:", 1)[1].split()[0])
            except (IndexError, ValueError):
                start = None
        elif "silence_end:" in line and start is not None:
            try:
                end = float(line.rsplit("silence_end:", 1)[1].split()[0])
            except (IndexError, ValueError):
                start = None
                continue
            spans.append((start, end))
            start = None
    return spans


def plan_segments(
    duration: float,
    silences: list[tuple[float, float]],
    target: float = CHUNK_TARGET_S,
    max_len: float = CHUNK_MAX_S,
    min_len: float = CHUNK_MIN_S,
) -> list[tuple[float, float]]:
    """Split [0, duration] into segments of at most max_len, cutting inside
    silence as close to `target` as possible. Falls back to a hard cut at
    max_len when a window contains no silence at all."""
    cuts = [(a + b) / 2 for a, b in silences]
    segments: list[tuple[float, float]] = []
    start = 0.0
    while duration - start > max_len:
        lo, hi = start + min_len, start + max_len
        window = [c for c in cuts if lo <= c <= hi]
        cut = min(window, key=lambda c: abs(c - (start + target))) if window else hi
        segments.append((start, cut))
        start = cut
    segments.append((start, duration))
    return segments


def slice_audio(path: str, segments: list[tuple[float, float]], out_dir: Path) -> list[Path]:
    """Cut `path` into 16 kHz mono wav files at the given segment boundaries,
    in a single decode pass. Returns the chunk paths in order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    times = ",".join(f"{end:.3f}" for _, end in segments[:-1])
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-i", path,
        "-vn", "-ac", "1", "-ar", "16000",
    ]
    if times:
        cmd += ["-f", "segment", "-segment_times", times]
        cmd += [str(out_dir / "%05d.wav")]
    else:
        cmd += [str(out_dir / "00000.wav")]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return sorted(out_dir.glob("*.wav"))


def build_plan(
    files: list[str],
    enabled: bool,
    tmp_root: Path,
    target: float = CHUNK_TARGET_S,
) -> list[dict]:
    """Return one entry per input file: {'file', 'duration', 'chunks', 'spans'}.
    Short files (or --no-chunk) pass through as a single chunk."""
    plan: list[dict] = []
    for i, f in enumerate(files):
        duration = probe_duration(f)
        if not enabled or duration is None or duration <= CHUNK_THRESHOLD_S:
            plan.append({
                "file": f, "duration": duration,
                "chunks": [f], "spans": [(0.0, duration or 0.0)],
            })
            continue
        segments = plan_segments(duration, detect_silences(f), target=target)
        chunks = slice_audio(f, segments, tmp_root / f"{i:04d}")
        if len(chunks) != len(segments):
            # ffmpeg landed on a different count; trust the files it wrote.
            segments = segments[: len(chunks)]
        print(
            f"▸ {Path(f).name}: {duration / 60:.1f} min → {len(chunks)} chunks",
            file=sys.stderr,
        )
        plan.append({
            "file": f, "duration": duration,
            "chunks": [str(c) for c in chunks], "spans": segments,
        })
    return plan


def transcribe(model, files: list[str], batch_size: int) -> list[str]:
    import torch
    with torch.no_grad():
        outs = model.transcribe(files, batch_size=batch_size, verbose=False)
    results = []
    for item in outs:
        if isinstance(item, list):
            item = item[0] if item else ""
        if hasattr(item, "text"):
            item = item.text
        if isinstance(item, tuple):
            item = item[0]
        results.append(str(item))
    return results


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Transcribe Lithuanian audio with the fine-tuned parakeet-tdt model",
    )
    ap.add_argument("files", nargs="+", help="Audio files to transcribe")
    ap.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"HF model id or local .nemo path (default: {DEFAULT_MODEL})",
    )
    ap.add_argument(
        "--lm", action="store_true",
        help="Enable beam search + token-level n-gram LM (best accuracy, ~10x slower)",
    )
    ap.add_argument(
        "--lm-path", default=None,
        help="Path to a local .arpa LM file (default: download from HF)",
    )
    ap.add_argument("--lm-alpha", type=float, default=0.3, help="LM weight (default 0.3)")
    ap.add_argument("--beam-size", type=int, default=4, help="Beam size (default 4)")
    ap.add_argument(
        "--device", default=None,
        help="'cuda' or 'cpu' (default: cuda if available)",
    )
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--json", action="store_true",
        help="Emit JSON lines to stdout (one object per file)",
    )
    ap.add_argument(
        "--no-chunk", action="store_true",
        help="Feed each file to the model whole. Loses large spans on long "
             "recordings — only useful for reproducing the old behaviour.",
    )
    ap.add_argument(
        "--chunk-seconds", type=float, default=CHUNK_TARGET_S,
        help=f"Target chunk length in seconds (default: {CHUNK_TARGET_S:.0f})",
    )
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model, device)

    if args.lm:
        lm_path = resolve_lm_path(args.lm_path, args.model)
        enable_lm(model, lm_path, beam_size=args.beam_size, alpha=args.lm_alpha)

    with tempfile.TemporaryDirectory(prefix="parakeet-chunks-") as tmp:
        plan = build_plan(
            args.files, not args.no_chunk, Path(tmp), target=args.chunk_seconds,
        )
        flat = [c for entry in plan for c in entry["chunks"]]

        print(
            f"▸ transcribing {len(args.files)} file(s) "
            f"({len(flat)} chunk(s)) on {device}...",
            file=sys.stderr,
        )
        t0 = time.time()
        chunk_texts = transcribe(model, flat, args.batch_size)
        elapsed = time.time() - t0
        print(f"▸ done in {elapsed:.1f}s", file=sys.stderr)

        # Reassemble per source file, streaming results as each one completes.
        pos = 0
        for entry in plan:
            n = len(entry["chunks"])
            parts = chunk_texts[pos:pos + n]
            pos += n
            segments = [
                {"start": round(a, 2), "end": round(b, 2), "text": t.strip()}
                for (a, b), t in zip(entry["spans"], parts)
            ]
            text = " ".join(t.strip() for t in parts if t.strip())
            if args.json:
                print(json.dumps({
                    "file": entry["file"],
                    "text": text,
                    "duration": entry["duration"],
                    "segments": segments,
                }, ensure_ascii=False), flush=True)
            else:
                print(f"{entry['file']}\t{text}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
