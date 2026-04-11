#!/usr/bin/env python3
"""Fine-tune parakeet-tdt-0.6b-v3 on Lithuanian.

Uses a raw PyTorch training loop instead of Lightning. Lightning's CUDA
graph toggling and checkpoint callbacks caused silent failures in
multiple runs — adapters ended up on CPU while the model was on CUDA,
checkpoints weren't saved, and the model output was silently corrupted.

The raw loop gives us:
- Verified per-epoch WER eval on dev clips
- Early stopping if the model starts diverging
- Reliable .nemo checkpoint saving after each epoch
- Direct control over the training forward pass

Run:
    python scripts/05_finetune.py \
        --config configs/finetune_lt.yaml \
        --model nvidia/parakeet-tdt-0.6b-v3 \
        --out-dir checkpoints/lt-ft-v1

Smoke test:
    python scripts/05_finetune.py \
        --config configs/finetune_lt.yaml \
        --model nvidia/parakeet-tdt-0.6b-v3 \
        --out-dir checkpoints/lt-smoke \
        --train-manifest data/manifests/smoke_test.json \
        --epochs 2 --max-steps 20
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

import torch
from omegaconf import DictConfig, OmegaConf


# ── WER eval ──────────────────────────────────────────────────────────

PUNCT_RE = re.compile(r"[\"\'\,\.\!\?\;\:\(\)\[\]\{\}\«\»\"\"\„\‟\‚\'\'\–\—\-]")
SPACE_RE = re.compile(r"\s+")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    s = PUNCT_RE.sub(" ", s)
    return SPACE_RE.sub(" ", s).strip()


def eval_wer(model, manifest_path: str, n_clips: int = 200, show: int = 3) -> float:
    """Normalized WER eval on GPU. Returns WER as a percentage."""
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

    was_training = model.training
    model.eval()
    with torch.no_grad():
        outs = model.transcribe(files, batch_size=8, verbose=False)
    if was_training:
        model.train()

    hyps_raw = []
    for item in outs:
        if isinstance(item, list):
            item = item[0] if item else ""
        if hasattr(item, "text"):
            item = item.text
        if isinstance(item, tuple):
            item = item[0]
        hyps_raw.append(str(item))

    refs = [_norm(r) for r in refs_raw]
    hyps = [_norm(h) for h in hyps_raw]
    wer = jiwer.wer(refs, hyps) * 100

    for i in range(min(show, len(refs))):
        print(f"    REF: {refs_raw[i][:75]}", flush=True)
        print(f"    HYP: {hyps_raw[i][:75]}", flush=True)
        print(flush=True)

    return wer


# ── numba / loss helpers ──────────────────────────────────────────────

def _numba_cuda_works() -> bool:
    try:
        import numba.cuda

        @numba.cuda.jit("void(float32[:])")
        def _noop(x):
            pass

        x = torch.zeros(1, device="cuda", dtype=torch.float32)
        _noop[1, 1](numba.cuda.as_cuda_array(x))
        return True
    except Exception as e:
        print(f"  numba CUDA smoke test failed: {e}", flush=True)
        return False


def maybe_swap_to_pytorch_tdt_loss(model) -> None:
    """Swap to pure-PyTorch TDT loss if numba CUDA doesn't work."""
    from omegaconf import open_dict as _od
    from nemo.collections.asr.losses.rnnt import RNNTLoss

    loss_cfg = model.cfg.get("loss", {})
    loss_name = loss_cfg.get("loss_name", "default")
    if loss_name not in ("tdt", "default"):
        return

    if _numba_cuda_works():
        print("  ✓ numba CUDA OK — keeping fast tdt loss", flush=True)
        return

    print("  numba broken — falling back to tdt_pytorch", flush=True)
    tdt_kwargs = loss_cfg.get("tdt_kwargs", {})
    durations = tdt_kwargs.get("durations", None)
    sigma = tdt_kwargs.get("sigma", 0.0)
    num_classes = model.joint.num_classes_with_blank - 1 - model.joint.num_extra_outputs
    kwargs = {"durations": durations, "sigma": sigma}

    new_loss = RNNTLoss(
        num_classes=num_classes, loss_name="tdt_pytorch",
        loss_kwargs=kwargs, reduction=model.cfg.get("rnnt_reduction", "mean_batch"),
    )
    model.loss = new_loss
    if getattr(model.joint, "fuse_loss_wer", False):
        model.joint.set_loss(new_loss)
    with _od(model.cfg):
        model.cfg.loss.loss_name = "tdt_pytorch"
        model.cfg.loss.tdt_pytorch_kwargs = kwargs
    print(f"  ✓ swapped to tdt_pytorch", flush=True)


# ── Main ──────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).resolve().parent.parent / "configs/finetune_lt.yaml")
    ap.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--train-manifest", type=Path, default=None)
    ap.add_argument("--val-manifest", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load config ───────────────────────────────────────────────
    cfg = OmegaConf.load(args.config)
    if args.train_manifest:
        cfg.data.train_ds.manifest_filepath = str(args.train_manifest)
    if args.val_manifest:
        cfg.data.validation_ds.manifest_filepath = str(args.val_manifest)
    if args.epochs is not None:
        cfg.train.max_epochs = args.epochs
    if args.batch_size is not None:
        cfg.data.train_ds.batch_size = args.batch_size

    print("▸ effective config:\n" + OmegaConf.to_yaml(cfg), flush=True)
    (args.out_dir / "effective_config.yaml").write_text(OmegaConf.to_yaml(cfg))

    # ── Load model ────────────────────────────────────────────────
    import nemo.collections.asr as nemo_asr

    model_arg = args.model
    if model_arg.endswith(".nemo") and Path(model_arg).exists():
        print(f"▸ loading local checkpoint: {model_arg}", flush=True)
        model = nemo_asr.models.ASRModel.restore_from(model_arg)
    else:
        print(f"▸ loading pretrained: {model_arg}", flush=True)
        model = nemo_asr.models.ASRModel.from_pretrained(model_arg)

    model = model.to("cuda")
    maybe_swap_to_pytorch_tdt_loss(model)

    # ── Set up data ───────────────────────────────────────────────
    model.setup_training_data(train_data_config=cfg.data.train_ds)
    print(f"  train: {cfg.data.train_ds.manifest_filepath}", flush=True)
    train_dl = model._train_dl

    val_manifest = str(cfg.data.validation_ds.manifest_filepath)
    print(f"  val:   {val_manifest}", flush=True)

    # ── Set trainable parameters ──────────────────────────────────
    # Config knob: `train` can list "encoder", "decoder", "joint".
    # Default trains everything except the preprocessor.
    train_parts = cfg.train.get("trainable", ["encoder", "decoder", "joint"])
    if isinstance(train_parts, str):
        train_parts = [train_parts]

    for p in model.parameters():
        p.requires_grad = False
    for part in train_parts:
        mod = getattr(model, part, None)
        if mod is None:
            print(f"  ! unknown trainable part: {part}", flush=True)
            continue
        for p in mod.parameters():
            p.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  trainable parts: {train_parts}", flush=True)
    print(f"  trainable: {n_params/1e6:.1f}M / {total_params/1e6:.1f}M "
          f"({100*n_params/total_params:.1f}%)", flush=True)

    # ── Optimizer + scheduler ─────────────────────────────────────
    lr = float(cfg.optim.get("lr", 1e-6))
    wd = float(cfg.optim.get("weight_decay", 1e-3))
    min_lr = float(cfg.optim.get("min_lr", 0.0))
    warmup_steps = int(cfg.optim.get("warmup_steps", 0))
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd,
                                  betas=(0.9, 0.98))
    print(f"  optimizer: AdamW lr={lr} wd={wd}", flush=True)
    if warmup_steps > 0 or min_lr > 0:
        print(f"  lr schedule: warmup_steps={warmup_steps} → cos decay to {min_lr}",
              flush=True)

    # ── Training config ───────────────────────────────────────────
    max_epochs = int(cfg.train.get("max_epochs", 5))
    accum_steps = int(cfg.train.get("accumulate_grad_batches", 8))
    clip_val = float(cfg.train.get("gradient_clip_val", 1.0))
    log_every = int(cfg.train.get("log_every_n_steps", 100))
    max_steps = args.max_steps  # None = no limit
    early_stop_delta = float(cfg.train.get("early_stop_wer_delta", 5.0))

    # Disable amp — RNN-T/TDT loss is numerically sensitive to bf16.
    # NeMo's RNNTLoss has force_float32 internally, but under autocast
    # the joint output may arrive in bf16 and cause gradient corruption.
    use_amp = False
    amp_dtype = torch.float32
    print(f"  amp: disabled (RNN-T loss needs fp32)", flush=True)

    # ── Training loop ─────────────────────────────────────────────
    baseline_wer = None
    best_wer = float("inf")
    best_epoch = -1
    global_step = 0

    def _freeze_bn(m):
        """Keep batch norm in eval mode during training.

        The pretrained encoder's batch norm running statistics are tuned
        to its training data distribution. Updating them with our
        Lithuanian fine-tuning data destroys the encoder representations
        and causes catastrophic forgetting — even with no parameter
        updates (just forward passes in train mode corrupt the model).
        """
        for mod in m.modules():
            if isinstance(mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                                torch.nn.BatchNorm3d)):
                mod.eval()

    for epoch in range(max_epochs):
        print(f"\n▸ === EPOCH {epoch}/{max_epochs} ===", flush=True)
        model.train()
        _freeze_bn(model)  # critical — prevents BN running stats corruption
        optimizer.zero_grad()
        epoch_loss = 0.0
        n_loss = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_dl):
            if max_steps and global_step >= max_steps:
                break

            signal, signal_len, transcript, transcript_len = batch
            signal = signal.to("cuda", non_blocking=True)
            signal_len = signal_len.to("cuda", non_blocking=True)
            transcript = transcript.to("cuda", non_blocking=True)
            transcript_len = transcript_len.to("cuda", non_blocking=True)

            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                encoded, encoded_len = model.forward(
                    input_signal=signal, input_signal_length=signal_len
                )
                decoder_out, target_len, _ = model.decoder(
                    targets=transcript, target_length=transcript_len
                )

                if getattr(model.joint, "fuse_loss_wer", False):
                    loss, _, _, _ = model.joint(
                        encoder_outputs=encoded, decoder_outputs=decoder_out,
                        encoder_lengths=encoded_len, transcripts=transcript,
                        transcript_lengths=target_len, compute_wer=False,
                    )
                else:
                    joint_out = model.joint(
                        encoder_outputs=encoded, decoder_outputs=decoder_out
                    )
                    loss = model.loss(
                        log_probs=joint_out, targets=transcript,
                        input_lengths=encoded_len, target_lengths=target_len,
                    )

                loss = loss / accum_steps

            loss.backward()
            epoch_loss += loss.item() * accum_steps
            n_loss += 1

            if (batch_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable, clip_val)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # LR schedule: linear warmup → cosine decay to min_lr.
                if warmup_steps > 0 or min_lr > 0:
                    import math
                    total_steps = max_epochs * max(1, len(train_dl) // accum_steps)
                    if global_step < warmup_steps:
                        factor = global_step / max(warmup_steps, 1)
                        cur_lr = lr * factor
                    else:
                        progress = (global_step - warmup_steps) / max(
                            total_steps - warmup_steps, 1
                        )
                        progress = min(max(progress, 0.0), 1.0)
                        cur_lr = min_lr + 0.5 * (lr - min_lr) * (
                            1 + math.cos(math.pi * progress)
                        )
                    for pg in optimizer.param_groups:
                        pg["lr"] = cur_lr

                if global_step % log_every == 0:
                    avg = epoch_loss / n_loss
                    elapsed = time.time() - t0
                    it_s = (batch_idx + 1) / elapsed
                    current_lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"  step {global_step}  batch {batch_idx+1}  "
                        f"loss={avg:.4f}  lr={current_lr:.2e}  {it_s:.1f} it/s",
                        flush=True,
                    )

        if max_steps and global_step >= max_steps:
            print(f"  reached max_steps={max_steps}", flush=True)

        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(n_loss, 1)
        print(f"  epoch {epoch} done in {elapsed:.0f}s  avg_loss={avg_loss:.4f}", flush=True)

        # ── Save checkpoint ───────────────────────────────────────
        epoch_nemo = args.out_dir / f"epoch{epoch:02d}.nemo"
        model.save_to(str(epoch_nemo))
        print(f"  ✓ saved {epoch_nemo} ({epoch_nemo.stat().st_size/1e9:.2f} GB)", flush=True)

        # ── WER eval ──────────────────────────────────────────────
        print(f"  WER eval (200 dev clips)...", flush=True)
        wer = eval_wer(model, val_manifest, n_clips=200)

        if baseline_wer is None:
            baseline_wer = wer
            print(f"  EPOCH {epoch}  WER: {wer:.2f}%  (baseline)", flush=True)
        else:
            delta = wer - baseline_wer
            marker = "▼" if delta < 0 else "▲"
            print(
                f"  EPOCH {epoch}  WER: {wer:.2f}% ({marker}{abs(delta):.2f}pp)",
                flush=True,
            )

        if wer < best_wer:
            best_wer = wer
            best_epoch = epoch
            import shutil
            shutil.copy2(epoch_nemo, args.out_dir / "best.nemo")
            print(f"  ✓ new best", flush=True)

        # ── Early stop ────────────────────────────────────────────
        if baseline_wer is not None and wer > baseline_wer + early_stop_delta:
            print(
                f"  ✗ WER {wer:.2f}% exceeds baseline+{early_stop_delta}pp "
                f"({baseline_wer + early_stop_delta:.2f}%) — stopping",
                flush=True,
            )
            break

    # ── Final ─────────────────────────────────────────────────────
    print(f"\n▸ done. best_epoch={best_epoch} best_WER={best_wer:.2f}%", flush=True)
    import shutil
    best_nemo = args.out_dir / "best.nemo"
    final_nemo = args.out_dir / "finetuned.nemo"
    if best_nemo.exists():
        shutil.copy2(best_nemo, final_nemo)
        print(f"▸ {final_nemo} ← best (epoch {best_epoch})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
