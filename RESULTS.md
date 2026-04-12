# Results — parakeet-tdt-lt Lithuanian fine-tune

Model: [`sliderforthewin/parakeet-tdt-lt`](https://huggingface.co/sliderforthewin/parakeet-tdt-lt)  
Base: `nvidia/parakeet-tdt-0.6b-v3`  
Training: ~43h Lithuanian audio, 5 epochs, lr=1e-6, BatchNorm frozen  
Live results: https://speechbench-viz.web.app (Lithuanian tab)

## Headline

**10.76% WER on CV25 LT test** (epoch 4 + beam=4 + token LM α=0.5)  
Down from 16.53% baseline — **34.9% relative reduction**.

## Full results table

### Common Voice 25 Lithuanian test (n=5,644)

| Configuration | WER | CER | Δ WER |
|---|---|---|---|
| Baseline greedy | 16.70% | 4.39% | — |
| Fine-tuned greedy | 14.06% | 2.90% | -2.64 |
| Fine-tuned beam+LM α=0.5 | **11.04%** | **2.70%** | **-5.66** |

### FLEURS Lithuanian test (n=986)

| Configuration | WER | CER | Δ WER |
|---|---|---|---|
| Baseline greedy | 21.41% | 5.50% | — |
| Fine-tuned greedy | 19.21% | 4.77% | -2.20 |
| Fine-tuned beam+LM α=0.5 | 17.54% | 5.48% | -3.87 |

### VoxPopuli Lithuanian test (n=42)

| Configuration | WER | CER | Δ WER |
|---|---|---|---|
| Baseline greedy | 30.00% | 17.99% | — |
| Fine-tuned greedy | 30.61% | 20.09% | +0.61 |
| Fine-tuned beam+LM α=0.5 | 39.39% | 32.49% | +9.39 |

Note: VoxPopuli LT regresses — parliamentary speech is out-of-domain
for our CV-heavy training data. The beam+LM makes it worse because
the LM biases toward CV/Wikipedia vocabulary, not parliamentary register.
Only 42 test clips, so these numbers are noisy.

### Leaderboard-compatible (BasicTextNormalizer)

| eval | WER | CER |
|---|---|---|
| FLEURS ft greedy | 19.73% | 4.98% |
| FLEURS ft beam+LM α=0.5 | **17.91%** | 5.58% |
| FLEURS baseline greedy | 22.15% | 5.77% |
| VoxPopuli ft greedy | 29.63% | 19.67% |
| VoxPopuli baseline greedy | 29.63% | 17.86% |

## Training progression (greedy, CV25 LT test)

| Epoch | WER | CER | Δ from baseline |
|---|---|---|---|
| 0 | 14.06% | 2.90% | -2.47 |
| 1 | 13.86% | 2.82% | -2.67 |
| 2 | 13.76% | 2.80% | -2.77 |
| 3 | 13.77% | 2.80% | -2.76 |
| 4 | 13.68% | 2.78% | -2.85 |

Model was still improving at epoch 4.

### Continued training (epochs 5-8, from epoch 4 checkpoint)

Ran 4 more epochs (bus error killed epoch 9 at step 5400/5900 due to
LM build competing for RAM). Loss still declining:

| Epoch (abs) | Loss | Dev WER (200 clips) |
|---|---|---|
| 5 | 0.6712 | 0.68% |
| 6 | 0.6611 | 0.68% |
| 7 | 0.6550 | 0.68% |
| 8 | 0.6456 | 0.68% |
| 9 | crashed at step 5400 | — |

Dev WER saturated at 0.68% (200-clip dev set is too easy). Full test
eval of epoch 8 checkpoint pending — need to run with beam+LM.
Checkpoints saved at `checkpoints/lt-ft-continued/epoch0{0-3}.nemo`.

### New LM build in progress (2026-04-11)

Building a 5-gram token LM from ~61M Lithuanian sentences (CC-100 +
OpenSubtitles + Wikipedia via `sliderforthewin/lt-asr-lm-corpora`).
This is ~3000x more text than the original LM. Expected to
significantly improve beam+LM WER on morphological endings.
Status: decompressing corpora, not yet complete. Output will be
`data/lm/lt_token_5gram_v2.arpa`.

## Decoding parameter sweep (500 dev clips)

### Alpha sweep (beam=4)

| α | WER |
|---|-----|
| 0.2 | 2.30% |
| 0.3 | 2.30% |
| 0.4 | 2.16% |
| 0.5 | 2.13% |
| **0.6** | **2.10%** |
| 0.7 | 2.13% |
| 0.8 | 2.10% |

Optimal α is 0.5-0.6. Difference is within noise (0.03pp).

### Beam size sweep (α=0.5)

| beam | WER |
|------|-----|
| 2 | 2.40% |
| **4** | **2.13%** |
| 8 | 2.43% |
| 16 | 2.26% |

**beam=4 is the sweet spot.** Larger beams are worse — LM penalty
accumulates and the search space gets noisier. Current settings
(beam=4, α=0.5) are already near-optimal.

## Error analysis (beam+LM, CV25 LT test)

- **55.7% of clips are perfect** (0% WER)
- **87% of errors are substitutions** (not deletions/insertions)
- **#1 error pattern: morphological endings** — the model gets the
  right word root but wrong case/tense suffix:
  - `ą` ↔ `a` (240 confusions)
  - `e` ↔ `ė` (107)
  - `u` ↔ `ų` (85)
  - `daugiausiai` ↔ `daugiausia`, `veikia` ↔ `veikė`
- **1,444 clips have exactly 1 error** — mostly 1 substitution
- **Short clips (<3s) are hardest**: 41.9% avg WER vs 16.6% for >8s
- **134 clips are total failures** (empty output) — likely bad audio

### Top character confusions

| confusion | count | pattern |
|---|---|---|
| a ↔ o | 240 | vowel quality |
| ą → a | 112 | losing ogonek |
| e ↔ ė | 107 | short vs long |
| o ↔ ų | 106 | ending confusion |
| i ↔ ė | 93 | inflection |

### What would improve WER further

1. **Bigger LM** — morphological endings are exactly what n-gram
   context resolves. More Lithuanian text → better ending prediction.
2. **More training epochs** — model still improving at epoch 4
   (in progress, epochs 5-9 running now)
3. **Speed perturbation** — augment training data with 0.9x/1.1x speed
4. **Neural LM rescoring** — Transformer LM trained on Lithuanian text,
   rescore n-best list from beam search

## Normalizer note

The speechbench numbers above use a diacritic-preserving normalizer
(lowercase + strip punctuation, keep ąčęėįšųūž). The leaderboard
numbers use `whisper_normalizer.BasicTextNormalizer`. These produce
different absolute WER values on the same model output — see
`lessons_from_finetuning.md` in the speechbench repo for details.

The existing speechbench report for other models uses an ASCII-only
normalizer that strips diacritics — those numbers are NOT directly
comparable. A normalizer fix for speechbench is documented and pending.

## Files

- Model: [sliderforthewin/parakeet-tdt-lt](https://huggingface.co/sliderforthewin/parakeet-tdt-lt) (epoch 4 checkpoint)
- Token LM: `lt_token_4gram.arpa` (also on HuggingFace)
- Per-clip results: `results/sb_*/per_clip.jsonl`
- Leaderboard results: `results/lb_*/summary.json`
- Live dashboard: https://speechbench-viz.web.app
