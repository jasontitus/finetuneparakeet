# Using the fine-tuned Lithuanian Parakeet-TDT model

A fine-tuned version of [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
specialized for Lithuanian ASR. Delivers **32% relative WER reduction**
on the Common Voice 25 Lithuanian test set versus the pretrained baseline.

## Results

On the full CV25 LT test set (5,644 clips, strict diacritic-preserving WER):

| Configuration | WER | CER | RTFx (RTX 3090) |
|---------------|-----|-----|-----|
| Pretrained baseline | 16.53% | 4.29% | 480× |
| **This model (greedy)** | **14.06%** | **2.90%** | 477× |
| **This model + beam + LM** | **11.23%** | **2.61%** | ~50× |

## Install

```bash
# Core dependencies
pip install 'nemo_toolkit[asr]>=2.0,<2.5' torch torchaudio

# Optional — only needed for beam+LM decoding (best accuracy)
pip install https://github.com/kpu/kenlm/archive/master.zip

# Optional — only needed if loading from HuggingFace Hub
pip install huggingface_hub
```

**Hardware:**
- **GPU (recommended):** any CUDA GPU with ≥4 GB VRAM. ~475× real-time
  on RTX 3090 / L4.
- **CPU:** works but ~15× real-time. 20 cores is plenty.

**WSL2 users:** See `WSL2_CUDA_ISSUES.md` — you need
`numba-cuda==0.15.1` and `LD_LIBRARY_PATH=/usr/lib/wsl/lib:...` or
the NeMo TDT loss kernel will crash.

## Quick start (greedy decoding, 14.06% WER)

```python
import nemo.collections.asr as nemo_asr

# Download + load from HuggingFace Hub
model = nemo_asr.models.ASRModel.from_pretrained(
    "sliderforthewin/parakeet-tdt-lt"
)
model = model.to("cuda")  # or "cpu"
model.eval()

# Transcribe one or more files
outs = model.transcribe(["audio.wav"])
print(outs[0].text)
```

That's it for the basic case. Works with WAV, FLAC, MP3, and any format
soundfile/librosa can decode. Any sample rate (the model resamples to
16 kHz internally).

## Full accuracy: beam search + n-gram LM (11.23% WER)

The model repo ships with `lt_token_4gram.arpa`, a 325 MB token-level
n-gram LM trained on 2.6M Lithuanian sentences (LT Wikipedia + CV25 +
VoxPopuli + FLEURS + shunyalabs). Enabling it drops WER by another 2.8
percentage points on the CV25 LT test set.

```python
import copy
import nemo.collections.asr as nemo_asr
from huggingface_hub import hf_hub_download
from omegaconf import open_dict

# 1. Load the model
model = nemo_asr.models.ASRModel.from_pretrained(
    "sliderforthewin/parakeet-tdt-lt"
).to("cuda").eval()

# 2. Download the LM from the same HF repo
lm_path = hf_hub_download(
    repo_id="sliderforthewin/parakeet-tdt-lt",
    filename="lt_token_4gram.arpa",
)

# 3. Switch to beam search + LM fusion
cfg = copy.deepcopy(model.cfg.decoding)
with open_dict(cfg):
    cfg.strategy = "maes"                # TDT beam strategy with LM support
    cfg.beam.beam_size = 4
    cfg.beam.return_best_hypothesis = True
    cfg.beam.ngram_lm_model = lm_path
    cfg.beam.ngram_lm_alpha = 0.3        # LM weight (0.2-0.4 works well)
model.change_decoding_strategy(cfg)

# 4. Transcribe
outs = model.transcribe(["audio.wav"])
print(outs[0].text)
```

Beam search is slower than greedy (~50× real-time vs ~475×) but the
accuracy gain is worth it for most use cases.

## Batch transcription

```python
model.transcribe(
    [f"clip{i}.wav" for i in range(100)],
    batch_size=8,     # bigger batch = faster, more GPU memory
    verbose=True,     # shows a progress bar
)
```

`transcribe()` streams through the list internally, so you can pass
thousands of files without loading them all at once.

## Standalone transcribe.py

For one-off use without writing Python:

```bash
python transcribe.py audio.wav                       # single file
python transcribe.py clip1.wav clip2.wav clip3.wav  # multiple files
python transcribe.py --lm audio.wav                  # with LM (slower, +2.8pp accuracy)
python transcribe.py --model /path/to/local.nemo --lm audio.wav  # local model
```

See `scripts/transcribe.py`.

## Advanced: what decoder strategies are available

Parakeet-TDT supports several decoding strategies. Switch via
`model.change_decoding_strategy(cfg)`:

| Strategy | LM support | Speed | Accuracy |
|----------|-----------|-------|----------|
| `greedy_batch` (default) | No | Fastest (~475×) | Good |
| `greedy` | No | Slow | Same as greedy_batch |
| `beam` | No | Medium | Marginal improvement over greedy |
| `maes` | **Yes** | Medium (~50×) | **Best** |
| `malsd_batch` | Yes (via different API) | Medium | Comparable to maes |

**Only `maes` supports the `ngram_lm_model` parameter directly.**
If you want LM fusion, use `strategy="maes"`.

## Tuning the LM weight (alpha)

`ngram_lm_alpha` controls how much the LM influences the beam. Good
defaults:

- **0.3** — works well on most Lithuanian audio (our validated value)
- **0.2** — if the LM is hurting accuracy on your domain
- **0.5** — only if you have a very in-domain LM and want aggressive
  fusion; otherwise the LM starts overriding valid acoustic predictions

For a new domain, sweep alpha on a held-out dev set:

```python
for alpha in [0.1, 0.2, 0.3, 0.4, 0.5]:
    cfg.beam.ngram_lm_alpha = alpha
    model.change_decoding_strategy(cfg)
    wer = eval_on_dev(model, dev_clips)
    print(f"alpha={alpha}: WER={wer:.2f}%")
```

**Don't tune on an "easy" dev subset.** We found all alphas gave
identical WER on our first 200 dev clips because the model was already
near-perfect on them. Pick a dev set that actually has errors to fix.

## Known limitations

- **Training data:** Only 19K clips (~43 hours) of read and
  parliamentary speech. Conversational speech, phone audio, and other
  domains may underperform.
- **Catastrophic clips:** ~2.6% of the CV25 test set (149 clips) still
  drift to wrong-language output (Cyrillic/Romanian/English gibberish).
  These clips appear to be out-of-distribution for the base model
  and fine-tuning didn't fully rescue them. LM fusion helps but
  doesn't eliminate.
- **Short clips (<3s):** Less context, higher error rate. LM fusion
  helps more here than on longer clips.
- **Domain shift:** Tested only on the same data distribution as
  training. Unknown performance on e.g. children's speech, heavy
  accents, or domain-specific terminology.

## Reproducing this model

See the repository README and `PLAN.md` for the full training recipe.
Short version:

1. Pull CV25 LT from Common Voice + VoxPopuli LT / FLEURS LT /
   shunyalabs LT via the scripts in `scripts/03_prepare_manifests.py`
2. Fine-tune with `scripts/05_finetune.py` for 5 epochs at lr=1e-6,
   encoder+decoder+joint all trainable, **BatchNorm frozen**,
   fp32 (no amp), raw PyTorch loop (no Lightning)
3. Build the token-level LM from LT Wikipedia + manifests via
   `scripts/08b_build_token_lm.py`
4. Eval with `scripts/11_eval_beam_lm.py`

**Critical gotcha:** BatchNorm running statistics must be frozen
during training, otherwise the pretrained encoder's normalization
stats get corrupted and the model catastrophically forgets Lithuanian.
See `FINETUNING_LESSONS.md` for the full story.

## License

The fine-tuned weights inherit NVIDIA's parakeet-tdt-0.6b-v3 license
(CC-BY-4.0). Training data is Common Voice 25 LT (CC0) + VoxPopuli LT
(CC0) + FLEURS LT (CC-BY-4.0) + shunyalabs LT (check source). The LM
is derived from LT Wikipedia (CC-BY-SA-4.0).

## Citation

If you use this model, please cite both the base parakeet and the
underlying datasets. See `PLAN.md` for details.
