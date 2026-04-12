---
language:
- lt
license: cc-by-4.0
tags:
- automatic-speech-recognition
- nemo
- parakeet
- tdt
- lithuanian
- fine-tuned
datasets:
- mozilla-foundation/common_voice_25_0
- facebook/voxpopuli
- google/fleurs
- shunyalabs/lithuanian-speech-dataset
metrics:
- wer
- cer
base_model: nvidia/parakeet-tdt-0.6b-v3
model-index:
- name: parakeet-tdt-lt
  results:
  - task:
      type: automatic-speech-recognition
      name: Speech Recognition
    dataset:
      name: Common Voice 25 Lithuanian
      type: mozilla-foundation/common_voice_25_0
      config: lt
      split: test
      args:
        language: lt
    metrics:
    - type: wer
      value: 9.01
      name: WER (beam + domain 5-gram LM α=0.5, BasicTextNormalizer)
    - type: wer
      value: 9.51
      name: WER (beam + domain 5-gram LM α=0.5)
    - type: cer
      value: 2.15
      name: CER (beam + domain 5-gram LM α=0.5)
    - type: wer
      value: 13.55
      name: WER (greedy)
    - type: cer
      value: 2.76
      name: CER (greedy)
  - task:
      type: automatic-speech-recognition
      name: Speech Recognition
    dataset:
      name: FLEURS Lithuanian
      type: google/fleurs
      config: lt_lt
      split: test
      args:
        language: lt
    metrics:
    - type: wer
      value: 15.87
      name: WER (beam + domain 5-gram LM α=0.5, BasicTextNormalizer)
    - type: wer
      value: 19.21
      name: WER (greedy)
    - type: cer
      value: 4.77
      name: CER (greedy)
---

# parakeet-tdt-lt — Lithuanian fine-tune of NVIDIA Parakeet TDT 0.6B v3

Fine-tuned version of [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) on ~43 hours of Lithuanian speech data. Achieves a **45.5% relative WER reduction** on Common Voice 25 Lithuanian test (16.53% → **9.01%** with beam search + domain 5-gram language model, BasicTextNormalizer).

## Results

| Configuration | CV25 LT WER | CV25 LT CER | FLEURS LT WER |
|---|---|---|---|
| Baseline (pretrained, greedy) | 16.53% | 4.29% | 22.15%* |
| Fine-tuned epoch 11 (greedy) | 13.55% | 2.76% | — |
| Fine-tuned + beam + domain 5-gram LM α=0.5 | **9.51%** | **2.15%** | — |
| Same, BasicTextNormalizer (leaderboard) | **9.01%** | **2.07%** | **15.87%** |

\* BasicTextNormalizer. Live results: [speechbench-viz.web.app](https://speechbench-viz.web.app)

## Usage

```python
import nemo.collections.asr as nemo_asr

# Greedy decoding
model = nemo_asr.models.ASRModel.from_pretrained("sliderforthewin/parakeet-tdt-lt")
transcriptions = model.transcribe(["audio.wav"])
```

### With beam search + language model (best quality)

```python
from omegaconf import open_dict
from huggingface_hub import hf_hub_download

model = nemo_asr.models.ASRModel.from_pretrained("sliderforthewin/parakeet-tdt-lt")

# Download the token-level LM
lm_path = hf_hub_download("sliderforthewin/parakeet-tdt-lt", "lt_token_4gram.arpa")

# Switch to beam search with LM fusion
decoding_cfg = model.cfg.decoding
with open_dict(decoding_cfg):
    decoding_cfg.strategy = "maes"
    decoding_cfg.beam.beam_size = 4
    decoding_cfg.beam.return_best_hypothesis = True
    decoding_cfg.beam.ngram_lm_model = lm_path
    decoding_cfg.beam.ngram_lm_alpha = 0.5
model.change_decoding_strategy(decoding_cfg)

transcriptions = model.transcribe(["audio.wav"])
```

## Training details

- **Base model**: nvidia/parakeet-tdt-0.6b-v3 (multilingual European, 25 languages)
- **Architecture**: Conformer encoder + Token-and-Duration Transducer (TDT)
- **Training data**: ~43h Lithuanian speech
  - Common Voice 25 LT train+validated+other (17.9h, 12.7k clips)
  - shunyalabs/lithuanian-speech-dataset (14.7h, 2.9k clips)
  - FLEURS LT (9.8h, 2.9k clips)
  - VoxPopuli LT (1.3h, 456 clips)
- **Recipe**: Raw PyTorch loop (no Lightning), AdamW lr=1e-6, 5 epochs, BatchNorm frozen to eval mode
- **Critical insight**: BatchNorm running statistics must be frozen — updating them with fine-tuning data destroys the pretrained encoder representations. See [lessons](https://github.com/jasontitus/finetuneparakeet/blob/main/WSL2_CUDA_ISSUES.md).
- **Language model**: 4-gram subword-token LM trained on training transcripts + Lithuanian Wikipedia (~240MB text), using the model's own SentencePiece tokenizer

## Reproduce

```bash
git clone https://github.com/jasontitus/finetuneparakeet.git
cd finetuneparakeet
bash scripts/gcp_eval.sh  # on a GCP VM with GPU
```

## Files

- `parakeet-tdt-lt.nemo` — NeMo checkpoint (epoch 11, best WER)
- `lt_domain_5gram.arpa` — Domain 5-gram token LM (Wikipedia + training manifests, recommended)
- `lt_token_4gram.arpa` — Original 4-gram token LM (smaller, still good)

## License

CC-BY-4.0 (same as the training data sources)
