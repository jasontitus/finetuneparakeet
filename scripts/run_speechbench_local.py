#!/usr/bin/env python3
"""Run speechbench evaluation locally (no GCP) for a model on Lithuanian datasets.

Imports speechbench's runner.run_job() directly to get the full perf
metrics (RTFx, latency percentiles, GPU peak memory) alongside
WER/CER with the fixed Unicode normalizer.

Usage:
    # From the finetuneparakeet directory, with speechbench on PYTHONPATH:
    PYTHONPATH=~/experiments/speechbench python scripts/run_speechbench_local.py \
        --model parakeet-tdt-lt \
        --datasets common_voice_25_lt fleurs_lt voxpopuli_lt \
        --out results/speechbench_full
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="speechbench model key")
    ap.add_argument("--datasets", nargs="+",
                    default=["common_voice_25_lt", "fleurs_lt", "voxpopuli_lt"])
    ap.add_argument("--sample-cap", type=int, default=0,
                    help="0 = full test set")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    # Import speechbench
    try:
        from speechbench.models import MODELS, make_model
        from speechbench.datasets import DATASETS, load
        from speechbench.runner import run_job
    except ImportError:
        print("ERROR: speechbench not on PYTHONPATH.", file=sys.stderr)
        print("  export PYTHONPATH=~/experiments/speechbench", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)

    model_spec = MODELS.get(args.model)
    if not model_spec:
        print(f"ERROR: model '{args.model}' not in MODELS registry", file=sys.stderr)
        print(f"  available: {list(MODELS.keys())[:10]}", file=sys.stderr)
        return 2

    model_obj = make_model(model_spec)
    print(f"▸ loading {args.model}...", flush=True)
    model_obj.load()

    for ds_key in args.datasets:
        ds_spec = DATASETS.get(ds_key)
        if not ds_spec:
            print(f"  ! unknown dataset: {ds_key}", flush=True)
            continue

        cap = args.sample_cap if args.sample_cap > 0 else (ds_spec.full_size or ds_spec.default_cap)
        print(f"\n▸ {ds_key} (cap={cap})...", flush=True)

        result = run_job(
            model=model_obj,
            dataset_spec=ds_spec,
            sample_cap=cap,
        )

        out_file = args.out / f"{ds_key}.json"
        with out_file.open("w") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        print(f"  WER:  {result.wer:.4f}")
        print(f"  CER:  {result.cer:.4f}")
        print(f"  RTFx: {result.rtfx_mean:.1f}")
        print(f"  GPU:  {result.gpu_peak_mem_mb:.0f} MB")
        print(f"  Wall: {result.wall_time_s:.1f}s")
        print(f"  → {out_file}", flush=True)

    model_obj.unload()
    print("\n▸ DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
