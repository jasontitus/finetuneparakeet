# Benchmarking parakeet-tdt-lt for speechbench + HuggingFace leaderboard

How to produce official, comparable numbers for `sliderforthewin/parakeet-tdt-lt`
on Lithuanian ASR datasets and get them published.

## Prerequisites

```bash
cd ~/finetuneparakeet
source .venv/bin/activate   # or create one: uv venv --python 3.11 .venv

# Core deps (skip if already installed from training)
pip install 'nemo_toolkit[asr]>=2.0,<2.5' jiwer datasets soundfile librosa whisper-normalizer

# WSL2 only — needed for numba TDT loss:
pip install 'numba-cuda==0.15.1'
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## What to benchmark

### Your best config (the headline number)

**Epoch 4 + beam search (beam=4) + token LM (α=0.5)** → 10.76% WER on CV25 LT.

This is the number you lead with everywhere. Always include it.

### Greedy baseline (for context)

Stock `nvidia/parakeet-tdt-0.6b-v3` greedy → 16.53% WER on CV25 LT.
Your fine-tuned model greedy → 13.68% WER on CV25 LT.

These show the improvement from fine-tuning alone vs fine-tuning + LM.

### Datasets

For speechbench: **CV25 LT, FLEURS LT, VoxPopuli LT** (the three
Lithuanian datasets in the speechbench table).

For the HuggingFace leaderboard: **FLEURS lt_lt** is the primary
multilingual benchmark. VoxPopuli lt is secondary.

## Step 1: speechbench numbers (your normalizer)

These use the diacritic-preserving normalizer from `04_eval.py` and
`11_eval_beam_lm.py`, and load data from local manifests.

```bash
# Build manifests (if not already done)
python scripts/03_prepare_manifests.py --datasets cv25_lt voxpopuli_lt fleurs_lt

# ── Fine-tuned model ──

# Greedy
python scripts/04_eval.py \
    --model sliderforthewin/parakeet-tdt-lt \
    --manifest data/manifests/cv25_lt_test.json \
    --out results/sb_ft_cv25_greedy

python scripts/04_eval.py \
    --model sliderforthewin/parakeet-tdt-lt \
    --manifest data/manifests/fleurs_lt_test.json \
    --out results/sb_ft_fleurs_greedy

python scripts/04_eval.py \
    --model sliderforthewin/parakeet-tdt-lt \
    --manifest data/manifests/voxpopuli_lt_test.json \
    --out results/sb_ft_voxpopuli_greedy

# Beam + LM (your best config)
python scripts/11_eval_beam_lm.py \
    --model sliderforthewin/parakeet-tdt-lt \
    --manifest data/manifests/cv25_lt_test.json \
    --lm data/lm/lt_token_4gram.arpa \
    --beam-size 4 --alpha 0.5 \
    --out results/sb_ft_cv25_beamlm

python scripts/11_eval_beam_lm.py \
    --model sliderforthewin/parakeet-tdt-lt \
    --manifest data/manifests/fleurs_lt_test.json \
    --lm data/lm/lt_token_4gram.arpa \
    --beam-size 4 --alpha 0.5 \
    --out results/sb_ft_fleurs_beamlm

python scripts/11_eval_beam_lm.py \
    --model sliderforthewin/parakeet-tdt-lt \
    --manifest data/manifests/voxpopuli_lt_test.json \
    --lm data/lm/lt_token_4gram.arpa \
    --beam-size 4 --alpha 0.5 \
    --out results/sb_ft_voxpopuli_beamlm

# ── Baseline model (for comparison column) ──

python scripts/04_eval.py \
    --model nvidia/parakeet-tdt-0.6b-v3 \
    --manifest data/manifests/cv25_lt_test.json \
    --out results/sb_baseline_cv25_greedy

python scripts/04_eval.py \
    --model nvidia/parakeet-tdt-0.6b-v3 \
    --manifest data/manifests/fleurs_lt_test.json \
    --out results/sb_baseline_fleurs_greedy

python scripts/04_eval.py \
    --model nvidia/parakeet-tdt-0.6b-v3 \
    --manifest data/manifests/voxpopuli_lt_test.json \
    --out results/sb_baseline_voxpopuli_greedy
```

Results land in `results/sb_*/summary.json`. Add these to the
speechbench web UI at `~/experiments/speechbench/web/public/results.json`
under `languages.lt.results` and deploy with `cd web && firebase deploy`.

## Step 2: leaderboard-comparable numbers (BasicTextNormalizer)

These use `whisper_normalizer.BasicTextNormalizer` — the exact
normalizer the Open ASR Leaderboard uses. Numbers are directly
comparable to any model on that leaderboard.

The script `12_eval_leaderboard.py` loads datasets from HuggingFace
(not local manifests) to guarantee identical test splits.

```bash
# ── Fine-tuned model ──

# Greedy
python scripts/12_eval_leaderboard.py \
    --model sliderforthewin/parakeet-tdt-lt \
    --dataset google/fleurs --config lt_lt --split test \
    --text-field transcription \
    --out results/lb_ft_fleurs_greedy

python scripts/12_eval_leaderboard.py \
    --model sliderforthewin/parakeet-tdt-lt \
    --dataset facebook/voxpopuli --config lt --split test \
    --text-field normalized_text \
    --out results/lb_ft_voxpopuli_greedy

# Beam + LM (your best — the headline number)
python scripts/12_eval_leaderboard.py \
    --model sliderforthewin/parakeet-tdt-lt \
    --dataset google/fleurs --config lt_lt --split test \
    --text-field transcription \
    --lm data/lm/lt_token_4gram.arpa --beam-size 4 --alpha 0.5 \
    --out results/lb_ft_fleurs_beamlm

python scripts/12_eval_leaderboard.py \
    --model sliderforthewin/parakeet-tdt-lt \
    --dataset facebook/voxpopuli --config lt --split test \
    --text-field normalized_text \
    --lm data/lm/lt_token_4gram.arpa --beam-size 4 --alpha 0.5 \
    --out results/lb_ft_voxpopuli_beamlm

# ── Baseline (for comparison) ──

python scripts/12_eval_leaderboard.py \
    --model nvidia/parakeet-tdt-0.6b-v3 \
    --dataset google/fleurs --config lt_lt --split test \
    --text-field transcription \
    --out results/lb_baseline_fleurs_greedy

python scripts/12_eval_leaderboard.py \
    --model nvidia/parakeet-tdt-0.6b-v3 \
    --dataset facebook/voxpopuli --config lt --split test \
    --text-field normalized_text \
    --out results/lb_baseline_voxpopuli_greedy
```

### CV25 with BasicTextNormalizer (no re-inference needed)

CV25 LT isn't on HuggingFace as a standard dataset, so
`12_eval_leaderboard.py` can't load it directly. But you already
have `per_clip.jsonl` from Step 1 with raw reference + hypothesis.
Re-normalize offline:

```python
import json
from whisper_normalizer.basic import BasicTextNormalizer
import jiwer

norm = BasicTextNormalizer()
refs, hyps = [], []
with open("results/sb_ft_cv25_beamlm/per_clip.jsonl") as f:
    for line in f:
        d = json.loads(line)
        r = norm(d["reference_raw"]).strip()
        h = norm(d["hypothesis_raw"]).strip()
        if r:
            refs.append(r)
            hyps.append(h)

wer = jiwer.wer(refs, hyps) * 100
cer = jiwer.cer(refs, hyps) * 100
print(f"CV25 LT beam+LM (BasicTextNormalizer): WER={wer:.2f}% CER={cer:.2f}%")
```

## Step 3: update HuggingFace model card

Copy `hf_model_card/README.md` to your HF repo. Update the
`model-index` YAML section with the leaderboard numbers from Step 2.

The `model-index` metadata makes results show up as benchmark
badges on your HuggingFace model page. Format:

```yaml
model-index:
- name: parakeet-tdt-lt
  results:
  - task:
      type: automatic-speech-recognition
    dataset:
      name: FLEURS Lithuanian
      type: google/fleurs
      config: lt_lt
      split: test
    metrics:
    - type: wer
      value: <YOUR_BEST_WER_HERE>
      name: WER (beam + LM α=0.5)
```

## Step 4: Open ASR Leaderboard submission

The leaderboard at `huggingface.co/spaces/hf-audio/open_asr_leaderboard`
auto-evaluates HuggingFace Transformers models. NeMo `.nemo` format
is NOT directly supported by their eval harness.

**Current best path:** Self-report with the `model-index` metadata
in your model card (Step 3). The numbers are computed with the same
normalizer (BasicTextNormalizer) and datasets (FLEURS, VoxPopuli)
the leaderboard uses, so they're directly comparable even if the
leaderboard doesn't run them automatically.

**Future path:** If NVIDIA adds NeMo support to the leaderboard's
eval harness, or if you convert the model to HF Transformers format
via `nemo2hf`, you can submit through the "Submit" tab and they'll
verify the numbers.

## Running on GCP

If the home machine isn't available, `vm_eval_final.sh` handles
everything on a GCP L4. **Critical: it pulls all models from GCS,
not HuggingFace.** Models must be pre-cached:

```
gs://safecare-maps-speechbench/corpora/models/parakeet-tdt-lt.nemo     ✓ uploaded
gs://safecare-maps-speechbench/corpora/models/parakeet-tdt-0.6b-v3.nemo ✓ uploaded
gs://safecare-maps-speechbench/corpora/lm/lt_token_4gram.arpa          ✓ uploaded
```

Never rely on HuggingFace downloads from GCP VMs — they stall 100%
of the time in this project.

## Expected results (from WSL runs)

| config | CV25 LT WER | CV25 LT CER |
|--------|-------------|-------------|
| Baseline greedy | 16.53% | 4.29% |
| Fine-tuned greedy | 13.68% | 2.78% |
| Fine-tuned beam+LM α=0.5 | **10.76%** | **2.57%** |

The GCP/leaderboard runs should reproduce these within ±0.1pp
(minor differences from audio decoding paths / floating point).
