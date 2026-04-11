#!/usr/bin/env python3
"""Minimal training loop to isolate catastrophic forgetting.

No Lightning, no callbacks, no checkpoints. Just:
1. Load pretrained model
2. Eval WER on 200 dev clips (baseline)
3. Run N training steps with raw PyTorch
4. Eval WER again (post-training)

This tells us whether the forgetting comes from the training itself
or from Lightning/NeMo's training infrastructure.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import torch

PUNCT_RE = re.compile(r"[\"\'\,\.\!\?\;\:\(\)\[\]\{\}\«\»\"\"\„\‟\‚\'\'\–\—\-]")
SPACE_RE = re.compile(r"\s+")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    s = PUNCT_RE.sub(" ", s)
    return SPACE_RE.sub(" ", s).strip()


def eval_wer(model, manifest_path: str, n_clips: int = 200) -> float:
    """Run normalized WER eval on GPU."""
    import jiwer

    clips = []
    with open(manifest_path) as f:
        for line in f:
            if line.strip():
                clips.append(json.loads(line))
                if len(clips) >= n_clips:
                    break

    files = [c["audio_filepath"] for c in clips]
    refs_raw = [c["text"] for c in clips]

    model.eval()
    with torch.no_grad():
        outs = model.transcribe(files, batch_size=8, verbose=False)

    hyps_raw = []
    for item in outs:
        if isinstance(item, list):
            item = item[0] if item else ""
        if hasattr(item, "text"):
            item = item.text
        if isinstance(item, tuple):
            item = item[0]
        hyps_raw.append(str(item))

    refs = [norm(r) for r in refs_raw]
    hyps = [norm(h) for h in hyps_raw]
    wer = jiwer.wer(refs, hyps) * 100

    # Print a few samples
    for i in range(min(3, len(refs))):
        print(f"    REF: {refs_raw[i][:70]}")
        print(f"    HYP: {hyps_raw[i][:70]}")
        print()

    return wer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-manifest", type=Path, required=True)
    ap.add_argument("--val-manifest", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--mode", choices=["decoder_joint", "adapter"], default="adapter")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"

    import nemo.collections.asr as nemo_asr

    print("▸ loading model...", flush=True)
    model = nemo_asr.models.ASRModel.from_pretrained(
        "nvidia/parakeet-tdt-0.6b-v3"
    )
    model = model.to("cuda")

    # --- Baseline eval ---
    print("\n▸ BASELINE eval:", flush=True)
    baseline_wer = eval_wer(model, str(args.val_manifest))
    print(f"  WER: {baseline_wer:.2f}%\n", flush=True)

    # --- Set up training ---
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    if args.mode == "adapter":
        from omegaconf import DictConfig
        adapter_cfg = DictConfig({
            "_target_": "nemo.collections.common.parts.adapter_modules.LinearAdapter",
            "in_features": 640, "dim": 32,
            "activation": "swish", "norm_position": "pre", "dropout": 0.0,
        })
        model.add_adapter(name="decoder:lt_adapter", cfg=adapter_cfg)
        model.add_adapter(name="joint:lt_adapter", cfg=DictConfig(dict(adapter_cfg)))
        model.set_enabled_adapters(enabled=True)
        # Move adapters to the same device as the model — they default
        # to CPU even if the model is on CUDA.
        for name, module in model.named_modules():
            if hasattr(module, "adapter_layer"):
                module.adapter_layer.to("cuda")
        print("  mode: adapter (decoder + joint, dim=32)", flush=True)
    else:
        for p in model.decoder.parameters():
            p.requires_grad = True
        for p in model.joint.parameters():
            p.requires_grad = True
        print("  mode: direct decoder+joint", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"  trainable params: {n_params:,}", flush=True)

    # --- Set up data ---
    from omegaconf import DictConfig as DC
    train_cfg = DC({
        "manifest_filepath": str(args.train_manifest),
        "sample_rate": 16000,
        "batch_size": 2,
        "shuffle": True,
        "num_workers": 2,
        "pin_memory": True,
        "max_duration": 20.0,
        "min_duration": 0.5,
        "trim_silence": False,
        "use_start_end_token": False,
        "is_tarred": False,
    })
    model.setup_training_data(train_data_config=train_cfg)
    train_dl = model._train_dl

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-3)
    print(f"  lr: {args.lr}", flush=True)

    # --- Training loop ---
    print(f"\n▸ training for {args.steps} steps...", flush=True)
    model.train()
    # Freeze batch norm — running stats update in train mode corrupts
    # the pretrained encoder representations.
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            m.eval()
    step = 0
    for batch in train_dl:
        if step >= args.steps:
            break

        # Move batch to GPU
        signal, signal_len, transcript, transcript_len = batch
        signal = signal.to("cuda")
        signal_len = signal_len.to("cuda")
        transcript = transcript.to("cuda")
        transcript_len = transcript_len.to("cuda")

        # Forward: encoder
        encoded, encoded_len = model.forward(
            input_signal=signal, input_signal_length=signal_len
        )

        # Forward: decoder
        decoder_out, target_len, _ = model.decoder(
            targets=transcript, target_length=transcript_len
        )

        # Forward: joint + loss
        # When fuse_loss_wer is enabled, the joint computes loss
        # internally. Otherwise we call joint then loss separately.
        if getattr(model.joint, "fuse_loss_wer", False):
            loss, _, _, _ = model.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoder_out,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=target_len,
                compute_wer=False,
            )
        else:
            joint_out = model.joint(
                encoder_outputs=encoded, decoder_outputs=decoder_out
            )
            loss = model.loss(
                log_probs=joint_out,
                targets=transcript,
                input_lengths=encoded_len,
                target_lengths=target_len,
            )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        step += 1
        print(f"  step {step}/{args.steps}  loss={loss.item():.4f}", flush=True)

    # --- Post-training eval ---
    print(f"\n▸ POST-TRAINING eval:", flush=True)
    post_wer = eval_wer(model, str(args.val_manifest))
    delta = post_wer - baseline_wer
    marker = "▼" if delta < 0 else "▲"
    print(f"  WER: {post_wer:.2f}% ({marker}{abs(delta):.2f}pp)", flush=True)

    print(f"\n▸ summary: {baseline_wer:.2f}% → {post_wer:.2f}%", flush=True)


if __name__ == "__main__":
    main()
