---
language:
  - lt
license: cc-by-4.0
base_model: nvidia/parakeet-tdt-0.6b-v3
tags:
  - automatic-speech-recognition
  - asr
  - speech
  - audio
  - lithuanian
  - nemo
  - parakeet
  - tdt
library_name: nemo
datasets:
  - mozilla-foundation/common_voice_17_0
  - facebook/voxpopuli
  - google/fleurs
metrics:
  - wer
  - cer
model-index:
  - name: parakeet-tdt-lt
    results:
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          name: Common Voice 25 LT (test)
          type: common_voice_25
          config: lt
          split: test
        metrics:
          - name: Test WER (greedy)
            type: wer
            value: 14.06
          - name: Test WER (beam+LM)
            type: wer
            value: 11.23
          - name: Test CER (beam+LM)
            type: cer
            value: 2.61
---

# parakeet-tdt-lt — Lithuanian fine-tune of parakeet-tdt-0.6b-v3

A Lithuanian ASR model, fine-tuned from NVIDIA's
[`parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).

**32% relative WER reduction** on Common Voice 25 Lithuanian test
(5,644 clips) versus the pretrained baseline.

| Model | WER | CER |
|-------|-----|-----|
| Pretrained (greedy) | 16.53% | 4.29% |
| **This model (greedy)** | **14.06%** | **2.90%** |
| **This model + beam + LM** | **11.23%** | **2.61%** |

## Quick start

```bash
pip install 'nemo_toolkit[asr]>=2.0,<2.5' torch
```

```python
import nemo.collections.asr as nemo_asr

model = nemo_asr.models.ASRModel.from_pretrained("jasontitus/parakeet-tdt-lt")
model = model.to("cuda")
model.eval()

text = model.transcribe(["audio.wav"])[0].text
print(text)
```

## Full accuracy (beam search + token-level n-gram LM)

This repo ships with a 325 MB token-level 4-gram LM trained on 2.6M
Lithuanian sentences (LT Wikipedia + the training manifests). Using
it drops WER from 14.06% to 11.23% on CV25 LT test.

```bash
pip install kenlm huggingface_hub
# (or: pip install https://github.com/kpu/kenlm/archive/master.zip)
```

```python
import copy
import nemo.collections.asr as nemo_asr
from huggingface_hub import hf_hub_download
from omegaconf import open_dict

model = nemo_asr.models.ASRModel.from_pretrained("jasontitus/parakeet-tdt-lt")
model = model.to("cuda").eval()

lm_path = hf_hub_download(
    repo_id="jasontitus/parakeet-tdt-lt",
    filename="lt_token_4gram.arpa",
)

cfg = copy.deepcopy(model.cfg.decoding)
with open_dict(cfg):
    cfg.strategy = "maes"
    cfg.beam.beam_size = 4
    cfg.beam.return_best_hypothesis = True
    cfg.beam.ngram_lm_model = lm_path
    cfg.beam.ngram_lm_alpha = 0.3
model.change_decoding_strategy(cfg)

text = model.transcribe(["audio.wav"])[0].text
print(text)
```

## Training

- **Base model:** `nvidia/parakeet-tdt-0.6b-v3` (627M params, Conformer
  encoder + RNN-T / TDT decoder)
- **Training data:** ~19,000 clips (~43 hours) from CV25 LT +
  VoxPopuli LT + FLEURS LT + shunyalabs LT
- **Recipe:** full model trainable (encoder + decoder + joint), but
  with **BatchNorm layers frozen to eval mode** — critical to prevent
  catastrophic forgetting of the pretrained acoustic representations
- **Optimizer:** AdamW, lr=1e-6, weight_decay=1e-3, 5 epochs, fp32
  (RNN-T loss is numerically sensitive to half precision)
- **Hardware:** RTX 3090, ~2.5 hours wall time
- **LM:** 4-gram token-level (over subword token IDs, not words) built
  from LT Wikipedia + training manifests, ~95M tokens

## Language model

The included `lt_token_4gram.arpa` is a **token-level** LM — it's
built over `chr(token_id + 100)` sequences, not over words. This
matches NeMo's TDT beam decoder interface (`tdt_beam_decoding.py`).
A word-level LM will silently fail to improve WER.

Training corpus:
- 2,647,722 cleaned sentences from `wikimedia/wikipedia/20231101.lt`
- 24,622 sentences from the ASR training manifests
- 95.7M total tokens after tokenization
- 7.4M 4-grams after min-count=2 filtering

## Intended use

- **Primary:** Lithuanian speech-to-text for read/literary and
  parliamentary audio (the training data distribution)
- **Expected OK:** news broadcasts, recorded lectures, audiobooks
- **Untested / expected weaker:** conversational speech, phone audio,
  children's speech, heavy accents, domain-specific technical vocabulary

## Limitations

- Only 43 hours of training data — not as robust as production-scale
  ASR models trained on 10,000+ hours
- ~2.6% of clips still produce unusable output (these were already
  catastrophic for the base model; fine-tuning partially recovered
  but didn't fully fix)
- No confidence scores
- No word-level timestamps (unless you enable `preserve_alignments`
  on the decoder)

## Citation and attribution

If you use this model, please cite both the base model and the
datasets:

```bibtex
@misc{parakeet-tdt-lt,
  title={parakeet-tdt-lt: Lithuanian fine-tune of parakeet-tdt-0.6b-v3},
  author={Titus, Jason},
  year={2026},
  howpublished={\url{https://huggingface.co/jasontitus/parakeet-tdt-lt}},
}
```

Base model: NVIDIA's
[`parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).

## License

Model weights: CC-BY-4.0 (inherited from base model).
Language model: Derived from LT Wikipedia (CC-BY-SA-4.0).
