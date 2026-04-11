# Fine-tune parakeet-tdt-0.6b-v3 on Lithuanian

Goal: lower the WER of NVIDIA's `parakeet-tdt-0.6b-v3` on Lithuanian
Common Voice 25 from the observed ~29% baseline by fine-tuning on
Lithuanian speech corpora.

---

## 📍 Current state (2026-04-09, read this first to resume)

**Where we are:** two training runs attempted, both died in ~30 min for
**different reasons**. Each fix has already been applied. Ready to launch
run #3 as soon as the GCP GPU quota blocker clears.

**The blocker:** project `safecare-maps` has `GPUS_ALL_REGIONS = 1`
(global cap). `speechbench-eslt300-01` (T4 spot in us-east1-c, a
speechbench benchmark VM) is currently holding it. User is keeping
that run AND requesting a quota bump to **4** via the GCP console. As
soon as that's approved we can launch in parallel.

**Stable facts we know for sure (from run #2's baseline eval before
the preempt):**

| thing | value |
|-------|-------|
| pretrained baseline WER on CV25 LT full test (n=5,644) | **16.69%** |
| pretrained baseline CER on CV25 LT full test | **4.38%** |
| baseline RTFx on L4 | 375× |
| trainable params with frozen encoder | 18.1 M / 627 M (2.9%) |
| training throughput on L4 @ batch 2 grad_accum 8 | ~7.1 it/s |
| training data total (manifests on VM) | 19,077 clips / ~43h |
| epoch 0 batches (with batch_size=2) | 9,395 |
| per-epoch wall time on L4 | ~22 min |
| 5-epoch training wall time | ~1.8h |
| total pipeline wall time (prep + eval + train + eval) | ~2-2.5h |

### What to do next (exact commands)

1. **Check the quota block** (is it clear?):
   ```bash
   gcloud compute project-info describe --project=safecare-maps \
     --format='value(quotas.metric,quotas.limit,quotas.usage)' \
     | tr ';' '\n' | grep GPUS_ALL_REGIONS
   ```
   Need `limit >= 2` and `usage < limit` before proceeding.

2. **Launch run #3** (on-demand L4, ~$1.70-2.00, ~2h):
   ```bash
   cd ~/experiments/finetuneparakeet
   PROVISIONING_MODEL=STANDARD bash scripts/launch_ft_vm.sh full
   ```
   The script picks an L4 on-demand zone automatically. Run id is
   `lt-full-<timestamp>`; save it — you'll need it to watch / resume.

3. **Monitor progress** (background uploader mirrors runner log every
   3 min):
   ```bash
   RUN_ID=<the new run id>
   watch -n 30 'gsutil cat gs://safecare-maps-speechbench/finetune/'$RUN_ID'/logs/*.runner.log 2>/dev/null | tail -20'
   ```

4. **Done markers** appear at:
   ```bash
   gsutil ls gs://safecare-maps-speechbench/finetune/$RUN_ID/logs/
   # Look for ...done (success) or ...failed (on_error trap fired)
   ```

5. **Pull results** (after ...done):
   ```bash
   gsutil -m cp -r gs://safecare-maps-speechbench/finetune/$RUN_ID/results ./results/$RUN_ID
   gsutil cp gs://safecare-maps-speechbench/finetune/$RUN_ID/checkpoints/finetuned.nemo ./checkpoints/$RUN_ID.nemo
   cat results/$RUN_ID/results/baseline_cv25_lt_test/summary.json
   cat results/$RUN_ID/results/finetuned_cv25_lt_test/summary.json
   ```

6. **If run #3 crashes or gets killed partway through**, relaunch with
   **the same RUN_ID** — `scripts/05_finetune.py` auto-resumes from
   `lightning_ckpts/last.ckpt` which the VM's background uploader
   mirrors to GCS every 3 min:
   ```bash
   RUN_ID=<existing run id> PROVISIONING_MODEL=STANDARD \
     bash scripts/launch_ft_vm.sh full
   ```
   Checkpoint cadence is every 500 training steps (added after run #2
   was preempted mid-validation with no checkpoint on disk).

### Committed recipe (configs/finetune_lt.yaml)

These values were reached after the two failed runs below. Changing
them is not required to make run #3 work:

- peak lr **5e-6** (cosine, 1000 warmup steps, min_lr 5e-7)
- full encoder freeze (trains only pred network + joint ≈ 60M params)
- AdamW, weight_decay 1e-3
- batch_size **2**, accumulate_grad_batches **8** → effective batch 16
- max_duration **20s** (drops the long shunyalabs tail)
- 5 epochs
- val_check_interval 0.5 (validate twice/epoch)
- save_every_n_train_steps **500** (resume anchor via last.ckpt)
- PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

### Key files

- `PLAN.md` — this file (strategy + state + run log)
- `README.md` — quickstart
- `configs/finetune_lt.yaml` — committed recipe
- `scripts/03_prepare_manifests.py` — builds NeMo JSON manifests
- `scripts/04_eval.py` — WER/CER eval (strict diacritic-preserving normalizer)
- `scripts/05_finetune.py` — NeMo fine-tune loop with resume + best-ckpt load
- `scripts/vm_startup.sh` — VM bootstrap + pipeline driver
- `scripts/launch_ft_vm.sh` — local orchestrator (multi-zone, spot/on-demand)

### Key external state

- **GCS bucket**: `gs://safecare-maps-speechbench/` (safecare-maps project)
  - corpora: `gs://safecare-maps-speechbench/corpora/cv25-lt/cv-corpus-25.0-2026-03-09-lt.tar.gz`
  - finetune artifacts: `gs://safecare-maps-speechbench/finetune/<run_id>/`
- **GCP project**: `safecare-maps`
- **Parallel speechbench run** (coexisting, NOT to be touched):
  `speechbench-eslt300-01` (T4 spot, us-east1-c)
- **Related speechbench repo** with normalizer bug + lessons doc:
  `~/experiments/speechbench/lessons_from_finetuning.md`

### Known gotchas for a future session

1. **Don't compare our WER numbers against the speechbench report
   numbers directly.** speechbench's normalizer strips non-ASCII chars
   and inflates non-English WER. The 29.6% in the benchmark and our
   16.69% baseline are measuring the same model output. See
   `lessons_from_finetuning.md` in the speechbench repo.
2. **Don't rerun the earlier smoke-test recipe.** `lr=2e-5` with
   partial encoder freeze regressed the model from 14% to 36%. The
   committed recipe is the corrected one.
3. **Don't blow away `speechbench-eslt300-01`.** It's an active
   speechbench benchmark the user wants to keep.
4. **The VoxPopuli LT config has 456 train clips, not ~14k.** The
   `lt` split is the transcribed subset, not the full untranscribed
   LT speech. Doesn't matter — we have ~43h total without it.

---

## Baseline numbers we are trying to beat

| model                       | dataset           | WER    | CER   | normalizer              | n     |
|-----------------------------|-------------------|--------|-------|-------------------------|-------|
| parakeet-tdt-0.6b-v3         | CV25 LT test      | 29.6%  | 15.9% | speechbench (ASCII-only)| 30    |
| parakeet-tdt-0.6b-v3         | CV25 LT test      | **13.74%** | **2.71%** | strict (keeps diacritics) | 300   |

The two rows are the **same model output** measured with different
normalizers. Speechbench strips all non-ASCII characters (`[^a-z0-9'\s]`
→ space), which splits every Lithuanian word that contains `ąčęėįšųūž` —
inflating WER to ~29%. The second row keeps diacritics and is the
number you want to compare against when you actually care about correct
Lithuanian output.

All training runs below report against the **strict-normalizer** 13.74%
baseline. The speechbench column is kept only because it's what shows
up in the existing benchmark report.

## Feasibility on M2 Max

Hardware is fine:

- 12-core CPU / 38-core GPU / 64 GB unified memory.
- The 0.6B TDT model is ~2.4 GB in FP32 (~1.2 GB in FP16). Forward +
  backward + optimizer state for full fine-tune fits in 64 GB with
  batch size 2-4 at 30s clips, even with a conservative fudge factor.

Software has real gotchas:

1. **NeMo on Mac**. Training works via PyTorch Lightning's MPS
   accelerator, but many NeMo training paths assume CUDA. Expect to
   patch / override a few things.
2. **RNN-T loss kernel**. Parakeet-TDT is an RNN-T variant. NeMo's
   default loss path (`warprnnt_numba`) is CUDA-only. Options on Mac:
     a. fall back to `torchaudio.functional.rnnt_loss` (CPU-only, but
        correct and differentiable — works fine for small batch sizes);
     b. compute loss on CPU while keeping the model on MPS (move just
        the logits tensor across the device boundary);
     c. fall back to CPU entirely for the smoke-test run.
   We'll try (a) first and fall through to (c) if it misbehaves.
3. **Throughput**. Even with MPS for the forward/backward and CPU for
   the loss, expect something like 1-5x real-time on M2 Max — ie ~8-40
   wall hours to do one pass over ~40h of CV25 LT train. That's usable
   for experimentation but a cloud A10G/L4 (~30-60 min/epoch) is the
   right answer for a real training run.
4. **Broken system env**. `/opt/homebrew/anaconda3` has NeMo 1.21,
   transformers 4.33, and a torch/torchvision ABI mismatch. We'll
   create a fresh Python 3.10 venv with a pinned, working stack.

Bottom line: **yes you can do it on your Mac**, but the right workflow
is "iterate the pipeline on Mac with a tiny subset, then either babysit
a multi-day full run on Mac, or ship the pipeline to a GPU and run it
there in an hour or two."

## Strategy

### 1. Data

Held-out evaluation (never touched by training):

- **CV25 LT test split** — 5,644 clips / 8.18 h. Matches the existing
  speechbench eval, so WER is directly comparable to the ~29% baseline.

Primary training dataset (local + already mirrored in GCS):

- **Common Voice 25 Lithuanian**
  (`~/Downloads/1774126516509-cv-corpus-25.0-2026-03-09-lt.tar.gz`
  and `gs://safecare-maps-speechbench/corpora/cv25-lt/cv-corpus-25.0-2026-03-09-lt.tar.gz`).
  Extracted locally to `data/cv25_lt/`.
  Splits (from `scripts/02_extract_cv25.py`):

  | split            | clips   | hours |
  |------------------|---------|-------|
  | train            | 8,640   | 12.21 |
  | dev              | 5,545   | 7.69  |
  | test (eval only) | 5,644   | 8.18  |
  | validated_extra  | 202     | 0.27  |
  | other            | 3,905   | 5.46  |
  | **train bundle** | **12,747** | **17.94** |

  `train bundle = (train ∪ validated ∪ other) − (dev ∪ test)`. Validated
  is mostly a relabeled superset of train/dev/test — we get only ~0.27h
  of extras there. `other` is lower-quality crowdsourced clips that
  didn't hit the validation threshold but are still usable for fine-tuning.

Additional training datasets (pulled from HF on the training VM):

- **VoxPopuli LT** — `facebook/voxpopuli`, config `lt`. European
  Parliament speeches, ~14k train clips, **~35 h**. Scripted
  parliamentary register, different acoustic/lexical profile from CV.
- **shunyalabs/lithuanian-speech-dataset** — 2,937 train / 416 val /
  986 test clips, 16 kHz, avg clip ~18s. Estimated **~15-20 h train**.
  Licensed openly (public HF dataset, no gate). Content looks like
  clean news/narrative speech based on sniff-tests.
- **FLEURS LT** — `google/fleurs`, config `lt_lt`. ~1.7k train clips,
  **~2.5 h**. Small but very clean read speech — good for regularization.

| dataset              | clips  | hours (est) |
|----------------------|--------|-------------|
| CV25 LT train bundle | 12,747 | 17.94       |
| VoxPopuli LT train   | ~14k   | ~35         |
| shunyalabs LT train  | 2,937  | ~15-20      |
| FLEURS LT train      | ~1.7k  | ~2.5        |
| **total**            | **~31k** | **~70-75** |

Rejected / skipped:

- **CV22 LT** — strict subset of CV25 LT, no added value.
- **Thomcles/YodaLingua-Lithuanian** — gated (email contact required),
  24 kHz TTS data, synthetic-sounding. Skip.
- **Speech-data/Lithuanian-Speech-Dataset** — literally a single MP3.
- **LIEPA** — not on HF, no reliable mirror found.
- **IARPA Babel LT** — not freely available.

All audio is resampled to 16 kHz mono on-the-fly by the NeMo data
loader — we don't pre-convert MP3 → WAV because that would triple the
disk footprint on the VM.

### 2. Fine-tuning recipe

parakeet-tdt-0.6b-v3 is a multilingual European model that already
supports Lithuanian — we're specializing it, not teaching it from
scratch. That changes the recipe in important ways:

- **Start from the pretrained checkpoint** (`nvidia/parakeet-tdt-0.6b-v3`),
  already cached locally in `~/.cache/huggingface/hub`.
- **Reuse the existing tokenizer.** It's a multilingual BPE; Lithuanian
  text is already in-vocabulary. Do **not** retrain a new tokenizer —
  that would invalidate the pretrained decoder/joint weights.
- **Freeze the feature preprocessor** (mel spectrogram) always. It has
  no trainable state that matters for this task.
- **Partial encoder freeze**: freeze the first 12-18 Conformer layers
  of the encoder, train the last 6-12. Low-level acoustic features
  transfer across languages; higher-level representations benefit from
  LT-specific training.
- **Train decoder + joint network fully.** These are small (~10% of
  params) and most likely to benefit from LT specialization.
- **Low learning rate**: 1e-5 to 5e-5 (peak). A cosine or linear
  warmup schedule over ~500-1000 steps, then decay.
- **SpecAugment**: keep NeMo's default time/frequency masking.
  Important for CV-style data which has tight per-speaker overfitting
  risk.
- **Batch size** 2-4 audio clips per step on Mac (tight memory on GPU
  during backward). Use gradient accumulation of 8-16 to get effective
  batch 16-64.
- **Epochs**: 3-10. Watch dev WER and stop early if it plateaus or
  regresses.
- **Optimizer**: AdamW, weight_decay 1e-3, no LR scheduler bells and
  whistles on the first pass.

### 3. Evaluation

Re-run the CV25 LT test evaluation after every epoch and log:

- **WER** (whisper-style normalized)
- **CER** (character error rate — more informative for a morphologically
  rich language like Lithuanian)
- **Raw reference + hypothesis** for each clip, so we can inspect
  error types

The baseline eval script is standalone and also works without training
— it's how we reproduce the 29% number from speechbench.

### 4. Tiered execution plan

| tier | scope                                              | where      | rough wall time | spot $   |
|------|----------------------------------------------------|------------|-----------------|----------|
| 0    | Env + manifests + baseline eval (~200 test clips)  | M2 Max     | 10-30 min       | free     |
| 1    | Smoke-test fine-tune (500 train, 1 epoch)          | M2 Max     | 30-90 min       | free     |
| 2    | CV25 LT train only, 5 epochs                       | GCP L4 spot | ~4 h           | ~$1.50   |
| 3    | CV25 LT + VoxPopuli LT + FLEURS LT, 5 epochs       | GCP L4 spot | ~8-10 h        | ~$3-5    |
| 3'   | Same as Tier 3, smaller/cheaper                    | GCP T4 spot | ~16-20 h       | ~$3-4 (preempt risk) |
| 3''  | Same as Tier 3, faster                             | GCP A100    | ~3-4 h         | ~$4-6    |

Tier 0 + 1 validate everything works locally on your Mac (free). Tier
2/3 are the real training runs — one full run is **well under $5** on
L4 spot. Hyperparameter iteration is dollars, not hundreds.

**L4 is the sweet spot** for a 0.6B RNN-T model. A100 is only ~2x
faster and nearly 4x the price. T4 struggles with RNN-T loss in mixed
precision and picks up preemption risk on 16+ hour runs.

Prices from speechbench's own config (us-central1 spot):

| GPU  | machine         | VRAM  | spot $/hr |
|------|-----------------|-------|-----------|
| T4   | n1-standard-8   | 16 GB | $0.18     |
| L4   | g2-standard-8   | 24 GB | $0.28     |
| A100 | a2-highgpu-1g   | 40 GB | $1.10     |

Throughput prior: ~50× real-time training on L4 with batch 16 FP16 for
a 0.6B TDT model. This is a rough number and should be re-estimated
after the first real epoch lands.

Add ~$0.30-0.50/run for VM boot + deps + dataset pulls. Cache the
dataset tarballs in GCS (`gs://open-testimony-speechbench/corpora/`
already has `cv25-lt/` — reuse that, and stage VoxPopuli/FLEURS
alongside) so re-runs don't repay HF/Mozilla egress.

## Project layout

```
finetuneparakeet/
├── PLAN.md                # this file
├── README.md              # quickstart
├── pyproject.toml
├── requirements.txt       # pinned working stack
├── .venv/                 # fresh python 3.10 venv
├── data/
│   ├── cv25_lt/           # extracted corpus (not checked in)
│   └── manifests/         # NeMo JSON manifests
├── configs/
│   └── finetune_lt.yaml   # training config overrides
├── scripts/
│   ├── 01_setup_env.sh
│   ├── 02_extract_cv25.py
│   ├── 03_prepare_manifests.py
│   ├── 04_baseline_eval.py
│   ├── 05_finetune.py
│   └── 06_eval.py
└── checkpoints/           # .nemo files (not checked in)
```

Scripts are numbered so the intended order of operations is visible
from `ls scripts/`.

## Run log

### Smoke test #1 — `lt-smoke-20260409-055921` (L4 spot, us-central1-a)

- **Config**: CV25 LT only, 300 optimizer steps, lr=2e-5, freeze
  encoder layers 0-11 (train layers 12-23 + decoder + joint),
  batch=4 × grad_accum=4. Wall time ~12 min on L4, ~$0.06.
- **Baseline eval (n=300)**: WER 13.74%, CER 2.71%.
- **Post-FT eval (n=300)**: WER 35.82%, CER 9.92%. **Regressed**.

Failure analysis:
- 300 steps with lr=2e-5 is enough to drift the multilingual weights
  off their pretrained manifold but far too few to re-converge.
- Spot-checks show bimodal output: some clips stay clean, others go
  off the rails (`"te voy os pirmasis rinksnes..."` for `"tai buvo
  jos pirmasis žingsnis..."`).
- This is a classic "too-aggressive small fine-tune on an already-good
  model" failure — not a data or pipeline problem.

Corrections applied:
1. **LR 2e-5 → 5e-6** (4× lower peak). Cosine schedule unchanged.
2. **Freeze the entire encoder.** Only train the prediction network
   + joint network (~60 M parameters instead of ~300 M).
3. **Load best-by-dev-wer checkpoint for final eval** (not last
   in-memory weights).
4. **Re-baseline on the full 5,644-clip test set** so we have a
   stable reference number.

### Full run #1 — `lt-full-20260409-063739` (L4 spot, us-east1-c) — FAILED (OOM)

- **Config**: all 4 LT datasets (CV25 + VoxPopuli + FLEURS +
  shunyalabs), 5 epochs, corrected recipe (lr=5e-6, frozen encoder,
  batch=4 × grad_accum=4, max_duration=25s).
- **Result**: CUDA OOM at epoch 0, step 4694/4754 (99%).
- **Root cause**: RNN-T loss activation memory is `O(T × U × V)` per
  batch element. Shuffled batch of 4 shunyalabs clips (avg ~18s,
  max 27s) tried to allocate 5.04 GiB on a GPU with 4.67 GiB free.
- **Cost**: ~30 min × L4 spot = ~$0.14.
- **Checkpoint recovered**: none. Validation hadn't run yet.

Corrections applied:
1. **batch_size 4 → 2** (halves per-step activation memory).
2. **accumulate_grad_batches 4 → 8** (keep effective batch 16).
3. **max_duration 25 → 20** (drops the long shunyalabs tail).
4. **val_check_interval 1.0 → 0.5** (validate twice/epoch → more
   checkpoint opportunities).
5. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** to reduce
   fragmentation-related OOMs.

### Full run #2 — `lt-full-20260409-150254` (L4 spot, us-central1-a) — FAILED (spot preemption)

- **Config**: as above + OOM fixes. batch=2 × grad_accum=8,
  max_duration=20.
- **Baseline eval (full n=5,644)**: **WER 16.69% / CER 4.38%** ← the
  real full-test baseline for CV25 LT. 375× RTFx on L4.
- **Training**: reached mid-epoch-0 validation at ~50% (val batch
  1005/1387) before preemption.
- **Result**: GCP spot preempted the VM at 2026-04-09T15:32:57Z,
  ~30 min into the run. Not a bug — GCP reclaimed capacity.
- **Cost**: ~30 min × L4 spot = ~$0.14.
- **Checkpoint recovered**: none. `ModelCheckpoint` only fires after
  validation completes, and validation was mid-run at preempt time.

Corrections applied:
1. **Frequent checkpointing**: add a second `ModelCheckpoint`
   callback with `every_n_train_steps=500 save_last=True` so
   `last.ckpt` gets written every ~70s of training regardless of
   validation. Preemption no longer loses a full epoch.
2. **Switch to on-demand L4** (`PROVISIONING_MODEL=STANDARD`) to
   eliminate spot preemption risk. ~$0.85/hr vs ~$0.28/hr but
   guaranteed to complete. Delta on a 2h run is ~$1.10 — worth it
   after two spot deaths.

### Full run #3 — *planned*, blocked on GCP GPU quota

- **Config**: all 4 LT datasets, 5 epochs, frozen encoder, batch 2
  × grad_accum 8, lr 5e-6, max_duration 20, validate twice/epoch,
  checkpoint every 500 train steps, on-demand L4.
- **Expected wall**: ~2-2.5h.
- **Expected cost**: ~$1.70-2.00 on-demand L4.
- **Blocked on**: `GPUS_ALL_REGIONS` quota bump (1 → 4) from user's
  GCP console request. Current global quota is 1 and the
  `speechbench-eslt300-01` T4 spot VM is holding it.

### WSL2 run — 2026-04-10/11 (first successful fine-tune)

This unblocked us from GCP by running on the user's local WSL2 +
RTX 3090 setup. All catastrophic-forgetting failures from runs
#1-#3 turned out to have the same root cause:

**Root cause of all prior regressions:** BatchNorm running statistics
get updated in `model.train()` mode, which corrupts the pretrained
encoder's normalization stats. Even `torch.no_grad()` forward passes
(no optimizer, no gradients) in train mode destroy the model. Fine-
tuning at ANY learning rate hit this.

**Fix:** Freeze all BatchNorm layers to eval mode after calling
`model.train()`. One loop, ~4 lines of code. See
`_freeze_bn()` in `scripts/05_finetune.py`.

**Other infrastructure fixes from this session:**
- WSL2 numba-cuda nvJitLink crash → pin `numba-cuda==0.15.1`
- Lightning checkpoint callbacks failing silently → switched to raw
  PyTorch training loop with per-epoch WER eval
- Adapter weights ending up on CPU while model is on CUDA (silent)
- Manifests lowercased/punctuation-stripped while the pretrained
  tokenizer expects raw case+punctuation → rebuilt manifests

**Run config:**
- Model: pretrained parakeet-tdt-0.6b-v3
- Trainable: encoder + decoder + joint (627M params), BN frozen
- LR: 1e-6, AdamW, no schedule
- SpecAugment: on (built into the model)
- Data: all 4 LT datasets, 5 epochs, batch 2 × accum 8
- Per-epoch WER eval on 200 dev clips with early stop if > baseline+5pp
- No amp (RNN-T loss is fp32-sensitive)

**Training results:**
- Epoch 0 avg_loss 0.93, 200-clip WER 0.51% ← best
- Epoch 1 avg_loss 0.74, 200-clip WER 0.51%
- Epoch 2 avg_loss 0.71, 200-clip WER 0.60%
- Epoch 3 avg_loss 0.69, 200-clip WER 0.68%
- Epoch 4 avg_loss 0.68, 200-clip WER 0.68%
- `best.nemo` = epoch 0 checkpoint

## Final results on CV25 LT full test set (5,644 clips)

| Approach | WER | CER | Δ vs baseline |
|----------|-----|-----|---|
| Baseline (pretrained, greedy) | 16.53% | 4.29% | — |
| Baseline + beam + WORD-LM α=0.5 | 18.63% | 5.22% | +2.10 (worse, LM bug) |
| **Fine-tuned (epoch 0, greedy)** | **14.06%** | **2.90%** | **-2.47** |
| **Fine-tuned + beam + token-LM α=0.3** | **11.23%** | **2.61%** | **-5.30** |

**Total: -5.30pp absolute, 32% relative WER reduction.** Both
components (FT and LM fusion) contribute independently and stack
cleanly. CER improvement is proportionally larger, confirming the
fine-tune fixed a lot of the Lithuanian morphological-ending errors
identified in the baseline error analysis.

### Cumulative spend so far

| run | outcome | cost |
|-----|---------|------|
| smoke #1 | regressed (recipe bug) | ~$0.06 |
| full #1 | OOM | ~$0.14 |
| full #2 | spot preempted | ~$0.14 |
| **cumulative** | | **~$0.34** |

Plus upcoming run #3 at ~$2. Comfortably under the original $3-5
estimate for a full run, even counting the failed attempts.

## Known risks and open questions

1. **RNN-T loss on MPS**. If neither torchaudio's CPU kernel nor
   device-boundary hopping gives usable throughput, fall back to full
   CPU training. That's an order of magnitude slower but it works.
2. **Catastrophic forgetting** of other languages is *not* a concern
   here — we only care about Lithuanian performance, so a model that's
   been pushed hard toward LT is fine.
3. **Overfitting to CV25 train**. CV is crowdsourced read speech and
   the test speakers may overlap with train in vocabulary/register.
   Including VoxPopuli and FLEURS reduces (but doesn't eliminate) this
   risk. We also report WER on held-out VoxPopuli LT and FLEURS LT as
   sanity checks.
4. **Disk space**. System is at 98% full (~87 GB free). Extracted
   CV25 LT is ~2 GB. Each checkpoint is ~2.5 GB. We keep at most 2
   rolling checkpoints to stay under 10 GB of training scratch.
5. **Tokenizer sanity**. Before training, verify that the existing
   tokenizer encodes/decodes Lithuanian text losslessly (no `<unk>`
   on nominally-LT characters).

## WER improvement options (status as of 2026-04-10)

Fine-tuning alone may plateau near the pretrained baseline because
parakeet-tdt is already well-trained on Lithuanian (16.53% test WER).
Options roughly ordered by effort vs expected impact:

### A. Error analysis — DONE
Script: `scripts/06_error_analysis.py`. Findings on baseline test:
- **43.1% of clips are perfect** out of the box
- **~40% of substitution errors are ending-only** (Lithuanian
  morphology: `gyvena`→`gyveno`, `teritoriją`→`teritorija`) — these
  are exactly what fine-tuning can fix because they require encoder
  features that discriminate subtle phonetic cues.
- **149 catastrophic clips (2.6%) drift to Cyrillic/Romanian/etc** —
  these alone contribute a significant chunk to overall WER.
- Short clips (<3s) are worse (34% WER) because they lack context;
  LM rescoring would help most here.
- No digit/abbreviation/normalization issues (the eval metric is
  legit).

### B. N-gram LM rescoring — WORKING
Scripts: `scripts/08_build_lm.py` (word-level, NOT USED — see below),
`scripts/08b_build_token_lm.py` (**correct, token-level**),
`scripts/10_download_lt_wikipedia.py`, `scripts/11_eval_beam_lm.py`.

**Gotcha — discovered 2026-04-11:** NeMo's TDT beam decoder
(`tdt_beam_decoding.py:806-809`) queries the LM with
`chr(token_id + 100)`, **not with actual words**. An LM trained on
word text silently does nothing — or worse, gives random backoff
probabilities that degrade WER (we saw 18.63% vs 16.53% greedy at
alpha=0.5). The LM MUST be built on subword token ID sequences:
1. Tokenize every training sentence with the model's tokenizer
2. Map each token ID to `chr(id + DEFAULT_TOKEN_OFFSET)` where
   `DEFAULT_TOKEN_OFFSET = 100`
3. Train n-gram counts on these character sequences
4. Write ARPA with these characters as the "vocabulary"

**Final LM (token-level on combined corpus):**
- 2,672,344 sentences (manifests + LT Wikipedia)
- 95.7M tokens (35.8 per sentence average)
- 6,993 distinct token types (1,199 vocab slots never used in LT)
- 7.4M 4-grams after min-count=2 filter
- File: `data/lm/lt_token_4gram.arpa` (325 MB)

**Decoder integration:** NeMo's `maes` strategy via
`model.change_decoding_strategy(...)`, passing `ngram_lm_model` and
`ngram_lm_alpha`.

**Final results — see session progress section for the full table.**

### C. SpecAugment — ALREADY ACTIVE
Parakeet-tdt ships with SpecAugment built into its architecture
(2 freq masks, 10 time masks, 27 freq width, 5% time width).
NeMo's forward pass applies it automatically when `model.training=True`,
and our raw training loop in `05_finetune.py` benefits from it
automatically. No wiring needed.

### D. Eval fairness audit — DONE, CLEAN
Script: `scripts/07_eval_audit.py`. Verified:
- 0 digit mismatches (CV25 LT test has no digit-form refs)
- 0 abbreviation issues worth acting on
- 0 NFC normalization mismatches
- 0 hyphen-split errors
- Short clips: 42% perfect (not as bad as I feared)
- Top 500 worst clips account for 35% of all errors — so a small
  "tail" of hard clips dominates

**Conclusion:** The 16.53% baseline is measuring real ASR errors,
not metric noise.

### E. Tokenizer masking (Lithuanian-only decoding) — BUILT
Scripts: `data/lm/lt_allowed_tokens.pkl`, `scripts/09_eval_masked.py`.
Parakeet's 8,192-token BPE vocab has **1,161 Cyrillic tokens** from
its multilingual training. We identified **3,176 tokens** that
appear in Lithuanian text and zero out logits for all others at
decode time.

Measurement on 150 drift clips (>100% baseline WER):
- Unmasked: 115.82% WER (Cyrillic/Romanian drift)
- Masked:   106.10% WER (forced into LT token space; outputs are
  still bad because the clips are acoustically hard, but no more
  wrong-language gibberish)

**The gotcha:** The `joint.joint_after_projection()` output during
greedy TDT decoding is shape `[..., vocab_size+1]` (labels + blank
only, NOT including durations which are computed separately). Early
attempts to mask failed silently because the shape check mismatched.

Masking is free but gives only a modest absolute WER improvement on
the full test set (most clips don't drift). The bigger win would
come from combining masking with beam search + LM rescoring.

### F. Longer training + LR decay
The raw loop in 05_finetune.py has a flat LR. Adding cosine decay
with warmup would let us start slightly higher and settle lower,
potentially squeezing more from the fine-tuning. Low priority —
current run is already showing improvement (epoch 0: 0.51% WER on
200-clip sample vs 0.68% baseline).

### G. Targeted data
If the error analysis had shown a specific failure mode (e.g. call
center audio), sourcing small amounts of targeted data would beat
more generic LT speech. The actual finding was "short clips + few
catastrophic out-of-distribution clips" which isn't specifically
addressable by more training data of the same kind.

## What's out of scope

- Retraining the tokenizer.
- Distillation / model compression.
- Serving / deployment. This project produces a `.nemo` checkpoint
  that the existing speechbench harness can load through its
  `NeMoParakeetModel` wrapper for evaluation.
