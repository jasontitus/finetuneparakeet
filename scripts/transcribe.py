#!/usr/bin/env python3
"""Transcribe audio with the fine-tuned Lithuanian parakeet-tdt model.

Simple standalone runner for the model published at
`jasontitus/parakeet-tdt-lt` on HuggingFace Hub. Handles downloading,
device placement, optional LM fusion, and printing results.

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

    # JSON output (for piping into other tools)
    python scripts/transcribe.py --json audio.wav
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path


DEFAULT_MODEL = "jasontitus/parakeet-tdt-lt"
DEFAULT_LM_FILENAME = "lt_token_4gram.arpa"


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
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model, device)

    if args.lm:
        lm_path = resolve_lm_path(args.lm_path, args.model)
        enable_lm(model, lm_path, beam_size=args.beam_size, alpha=args.lm_alpha)

    print(f"▸ transcribing {len(args.files)} file(s) on {device}...", file=sys.stderr)
    t0 = time.time()
    texts = transcribe(model, args.files, args.batch_size)
    elapsed = time.time() - t0
    print(f"▸ done in {elapsed:.1f}s", file=sys.stderr)

    for path, text in zip(args.files, texts):
        if args.json:
            print(json.dumps({"file": path, "text": text}, ensure_ascii=False))
        else:
            print(f"{path}\t{text}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
