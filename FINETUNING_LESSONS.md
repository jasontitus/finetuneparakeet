# Fine-tuning NeMo ASR models — lessons learned

A working-doc for Claude (and humans) picking up a NeMo ASR fine-tuning
task. This is focused on the things that **burned wall time** on this
project — the non-obvious bugs, the silent failures, the dead ends.
Read this before starting a new fine-tune on parakeet/Canary/etc.

Tested specifically against `nvidia/parakeet-tdt-0.6b-v3` fine-tuned on
Lithuanian (CV25 LT + VoxPopuli + FLEURS + shunyalabs). Most of it
generalizes to any NeMo RNN-T / TDT / CTC model.

## TL;DR — the one fix that matters

**Freeze BatchNorm to eval mode before every training step.**

```python
model.train()
for m in model.modules():
    if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
        m.eval()
```

Parakeet's Conformer encoder has 24 BatchNorm1d layers in its
convolutional modules. When `model.train()` is True, forward passes
update their running mean/variance — even with `torch.no_grad()` and
no backward pass. The pretrained stats are tuned to the multilingual
training distribution; your Lithuanian (or whatever) data shifts the
running stats and corrupts ALL downstream encoder representations.

**The symptom is catastrophic forgetting regardless of LR:**
- lr=2e-5 → 14% WER → 36% WER after a few hundred steps
- lr=5e-6 → 16.5% → 107%+ after one epoch
- lr=1e-8 → 0.7% → 101% after 100 steps
- Even `torch.no_grad()` forward passes in train mode → corruption

You will spend days chasing learning rate, adapters, loss functions,
Lightning bugs, and gradient accumulation before realizing BN is the
problem. Skip that. Just freeze BN first.

## Recipe that works (tested, 32% relative WER reduction)

```python
import torch
import nemo.collections.asr as nemo_asr

model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
model = model.to("cuda")

# Train everything except the preprocessor
for p in model.parameters(): p.requires_grad = False
for p in model.encoder.parameters(): p.requires_grad = True
for p in model.decoder.parameters(): p.requires_grad = True
for p in model.joint.parameters(): p.requires_grad = True

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-6, weight_decay=1e-3, betas=(0.9, 0.98),
)

model.setup_training_data(train_data_config=train_cfg)
train_dl = model._train_dl

# Critical: freeze BN after every model.train() call
def freeze_bn(m):
    for mod in m.modules():
        if isinstance(mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            mod.eval()

for epoch in range(5):
    model.train()
    freeze_bn(model)  # MUST come after model.train()
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_dl):
        signal, signal_len, transcript, transcript_len = [x.to("cuda") for x in batch]

        # Full forward — preprocessor + encoder (includes SpecAugment auto)
        encoded, encoded_len = model.forward(
            input_signal=signal, input_signal_length=signal_len,
        )
        decoder_out, target_len, _ = model.decoder(
            targets=transcript, target_length=transcript_len,
        )

        # Joint + loss. Parakeet uses fuse_loss_wer=True, so loss is
        # computed inside the joint forward, not via model.loss().
        if getattr(model.joint, "fuse_loss_wer", False):
            loss, _, _, _ = model.joint(
                encoder_outputs=encoded, decoder_outputs=decoder_out,
                encoder_lengths=encoded_len, transcripts=transcript,
                transcript_lengths=target_len, compute_wer=False,
            )
        else:
            joint_out = model.joint(
                encoder_outputs=encoded, decoder_outputs=decoder_out,
            )
            loss = model.loss(
                log_probs=joint_out, targets=transcript,
                input_lengths=encoded_len, target_lengths=target_len,
            )

        (loss / 8).backward()  # gradient accumulation

        if (batch_idx + 1) % 8 == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0,
            )
            optimizer.step()
            optimizer.zero_grad()

    model.save_to(f"epoch{epoch:02d}.nemo")
    # ... eval here, save best ...
```

**Key parameters that worked:**
- `lr=1e-6` AdamW — conservative but works. `5e-6` also works with BN frozen.
- No LR schedule (flat LR is fine for 5 epochs)
- `batch_size=2`, `accumulate_grad_batches=8` → effective batch 16
- `max_duration=20s`, `min_duration=0.5s`
- **No AMP** — RNN-T / TDT loss is fp32-sensitive. bf16 autocast corrupts gradients in subtle ways and degrades WER by ~5-10pp.
- 5 epochs is plenty. Epoch 0 was often the best; later epochs overfit.
- **SpecAugment is built into parakeet**, applied automatically in train mode. Don't add it manually.

## Don't use Lightning

Spent hours debugging Lightning. Failure modes observed:
- Silent checkpoint non-saving (`ModelCheckpoint` callbacks ran but wrote nothing)
- CUDA graph enable/disable around validation corrupted model state
- Adapter modules added after `model.to("cuda")` ended up on CPU; Lightning silently transferred the rest of the model but not the adapters
- `trainer.fit()` with per-epoch eval via callbacks was fragile
- Debugging info buried under numba warning spam (run with `PYTHONWARNINGS=ignore` and `python -u` for usable logs)

**Raw PyTorch training loops are simpler to debug and reason about.**
The full working loop for this project is ~250 lines in `scripts/05_finetune.py`.

## Manifests must match tokenizer format

Parakeet's SentencePieceBPE tokenizer was trained on mixed-case text with
punctuation. Don't pre-normalize the training manifests:

```python
# WRONG — lowercases and strips punctuation
text = normalize_text(raw_sentence)  # e.g. "grįžęs į lietuvą stojo..."

# RIGHT — NFC normalize + collapse whitespace only
text = unicodedata.normalize("NFC", raw_sentence).strip()  # "Grįžęs į Lietuvą, stojo..."
```

Feeding lowercase tokens to a mixed-case tokenizer pushes the model into
degenerate patterns (we had a run collapse into Cyrillic entirely because
lowercase LT tokens match Cyrillic lowercase subwords in the multilingual
vocab).

**Eval-time normalization is separate:** the eval script normalizes
BOTH reference and hypothesis equally (`04_eval.py` → lowercase + strip
punctuation), so WER comparison stays fair regardless of how the raw
data was stored.

## Verify your eval before the full run

**Always:**
1. Run a smoke test (10-20 optimizer steps on ~50 clips) that exercises
   the full pipeline including checkpoint saving and WER eval
2. Confirm checkpoints are actually on disk after the smoke test
3. Confirm the WER eval result is a sensible number (not NaN, not
   "100%" suggesting model corruption)
4. Only THEN start the full training run

I had two separate runs on different machines fail silently because
the ModelCheckpoint callbacks weren't writing. A smoke test would
have caught this in a minute. Instead it took 4+ hours of training
before we noticed.

## WER eval gotchas

**Small dev subsets are misleading.** The first 200 clips of CV25 LT
dev gave 0.68% WER at baseline; the full 5,545-clip dev gave 10.16%;
the full 5,644-clip test gave 16.53%. Always tune and report on the
full test/dev, not a head-of-file subset.

**NeMo's in-training WER does NOT normalize.** Its `val_wer` metric
compares raw tokenized output against raw reference text, so case
and punctuation differences inflate it. You may see `val_wer=78%`
during training while the model is actually fine. Run a separate
normalized eval (`scripts/04_eval.py`) with `model.transcribe()` to
get the real number.

**CER is the leading indicator for morphological errors.** Lithuanian
has rich noun/verb inflections (`gyvena` / `gyveno` / `gyvenau` / ...).
CER drops proportionally more than WER when fine-tuning fixes ending
confusion. Track both.

## WSL2 numba-cuda issue

`numba-cuda > 0.15.1` (as of 2026-04) crashes on WSL2 during kernel
compilation with `nvJitLink ERROR_INTERNAL (6) ... nvvmAddNVVMContainerToProgram`.

**Fix:**
```bash
uv pip install numba-cuda==0.15.1
```

Also needs `LD_LIBRARY_PATH` to point to the WSL2 driver proxy first:
```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}
```

Otherwise you'll see Issues 1-4 from `WSL2_CUDA_ISSUES.md`:
- `libnvvm.so cannot be found`
- Empty device list → `IndexError` during TDT loss
- nvJitLink crash during kernel compile

**Fallback** (if you really can't pin numba-cuda): use NeMo's pure
PyTorch TDT loss. The model ships with both:
```python
# After loading the model:
from nemo.collections.asr.losses.rnnt import RNNTLoss
new_loss = RNNTLoss(
    num_classes=model.joint.num_classes_with_blank - 1 - model.joint.num_extra_outputs,
    loss_name="tdt_pytorch",
    loss_kwargs={"durations": [0,1,2,3,4], "sigma": 0.02},
    reduction=model.cfg.get("rnnt_reduction", "mean_batch"),
)
model.loss = new_loss
# CRITICAL: also update the joint's cached reference
if model.joint.fuse_loss_wer:
    model.joint.set_loss(new_loss)
```
`tdt_pytorch` is ~3-4× slower than the numba version but avoids
all WSL2 issues. See `maybe_swap_to_pytorch_tdt_loss()` in
`scripts/05_finetune.py`.

## LM rescoring — the sneaky one

NeMo's TDT beam decoder supports n-gram LM fusion via `ngram_lm_model`.
The expected input is an ARPA (or binary KenLM) file.

**Gotcha: the LM is queried with TOKEN IDs, not words.**

In `tdt_beam_decoding.py` line ~806:
```python
def compute_ngram_score(self, current_lm_state, label):
    if self.token_offset:
        label = chr(label + self.token_offset)
    else:
        label = str(label)
    ...
```

`DEFAULT_TOKEN_OFFSET = 100`. The decoder calls
`lm.BaseScore(state, chr(token_id + 100), new_state)` — it's treating
each subword token as a single "character" in the LM vocabulary.

**An LM trained on actual word text silently does nothing (or worse,
returns random backoff probabilities) because the tokens being queried
never appear in its vocabulary.** I spent hours with a word-level LM
getting 18.63% WER (worse than 16.53% greedy) before finding this.

**The correct LM-building recipe:**
1. Tokenize each training sentence with the model's tokenizer
2. Map each token ID to `chr(id + 100)` — gives a sequence of
   single-character "tokens"
3. Count n-grams over those sequences
4. Write ARPA with these characters as the vocabulary

See `scripts/08b_build_token_lm.py`. The key difference from a
normal word-level LM:

```python
# WRONG — word-level LM, silently ignored by NeMo's beam decoder
tokens = sentence.split()

# RIGHT — token-level LM using the model's tokenizer + offset
ids = tokenizer.text_to_ids(sentence)
tokens = [chr(tid + 100) for tid in ids]
```

**For corpus:** LT Wikipedia via `datasets.load_dataset("wikimedia/wikipedia", "20231101.lt")`
gives ~211K articles → ~2.6M sentences → ~95M tokens after tokenization.
That's a proper LM training set. Our original 24K-sentence LM was too
small for reliable 4-gram statistics.

**Using the LM at inference:**
```python
from omegaconf import open_dict
import copy

cfg = copy.deepcopy(model.cfg.decoding)
with open_dict(cfg):
    cfg.strategy = "maes"  # the ONLY TDT strategy that supports LM fusion
    cfg.beam.beam_size = 4  # 4-8 is fine; larger beams rarely help
    cfg.beam.ngram_lm_model = "data/lm/lt_token_4gram.arpa"
    cfg.beam.ngram_lm_alpha = 0.3  # 0.3 was best in our sweep
model.change_decoding_strategy(cfg)
```

**Alpha tuning:** Values of 0.2-0.4 worked well. 0.5 started to
override the acoustic model and degrade WER. Sweep on a harder dev
subset, not the easy head of the dev set.

## Don't tune on an easy subset

200-clip "easy head" subsets gave identical WER for every alpha
(0.0 through 1.0) because the model was already near-perfect on
those specific clips. Tune on either:
- The full dev set (slow but accurate)
- A random sample across the dev set
- The clips with highest baseline WER (tests where LM matters most)

I wasted ~15 min on an alpha sweep that told me nothing because all
alphas reported the same 0.68% WER.

## Tokenizer masking for language restriction

Parakeet's 8,192-token vocab is multilingual: ~1,161 tokens are Cyrillic,
plus more from other Latin-alphabet languages. You can force LT-only
output by zeroing out non-LT tokens from the joint's logits:

1. Tokenize all your LT training text, collect the set of token IDs
   that appear → gets you ~3,176 "allowed" token IDs (out of 8,192)
2. Monkey-patch `joint.joint_after_projection` to `masked_fill` non-allowed
   positions with a very negative value BEFORE the beam/argmax

**Gotcha:** The joint's output during decode is shape `[..., vocab_size + 1]`
(labels + blank), **NOT** `[..., vocab_size + 1 + num_durations]`. Duration
logits are computed separately elsewhere. Early attempts to mask failed
silently because the shape check didn't match.

See `scripts/09_eval_masked.py`. In practice, LM rescoring subsumes
masking (the LM assigns near-zero probability to non-LT sequences),
so masking alone isn't worth the complexity. It's useful as a
debugging technique when you're trying to pin down drift issues.

## Debugging catastrophic forgetting — systematic approach

When the model's output goes from Lithuanian to gibberish after a
fine-tuning step, walk the symptoms:

1. **Is it deterministic?** Run 1 training step, then transcribe the
   same 3 easy clips. If broken → fundamental bug. If 100-step threshold
   → overfitting or LR issue.

2. **Is BN the problem?** Do N forward passes in `model.train()` mode
   with `torch.no_grad()` (NO backward, NO optimizer). If the model still
   breaks → BN is mutating state. Freeze BN.

3. **Is it device mismatch?** Print `next(mod.parameters()).device` for
   every major submodule (encoder, decoder, joint, loss, any adapters).
   Everything must be on the same device. Adapters added dynamically
   are especially prone to being on CPU.

4. **Is it autocast?** Run once with and once without `torch.autocast`.
   If bf16 breaks it, RNN-T loss can't handle half precision. Use fp32.

5. **Is it Lightning?** Strip out Lightning, run a minimal raw PyTorch
   loop. If the minimal loop works, something in Lightning's training
   loop is corrupting state. Stick with raw PyTorch.

6. **Is it gradient accumulation?** Run with accum=1. If that fixes it,
   you're misbalancing the loss scaling.

**The isolation test that found BN was:** literally just `model.train()`,
100 `torch.no_grad()` forward passes, then `model.eval()` and
`model.transcribe()`. Output went from perfect Lithuanian to English
gibberish. No optimizer, no backward, no gradients — pure forward.
That's the signal for "some Module is mutating state in train mode".

## Evaluation workflow that actually catches bugs

```
1. Baseline greedy eval on full test    → reference number
2. Train N steps with BN frozen
3. Eval trained model greedy on full test → is it better or worse than baseline?
4. If worse: STOP and diagnose (usually BN, device, or normalization)
5. If better: continue training
6. Track both WER and CER per epoch
7. Final: eval best checkpoint with beam+LM for the stacked result
```

**Always save per-clip outputs (`per_clip.jsonl`)** so you can diff
greedy vs beam+LM, diff baseline vs fine-tuned, and run error analysis
(`scripts/06_error_analysis.py`) to understand what's actually changing.

## Numbers from this project

| Stage | WER (CV25 LT test, 5644 clips) | CER |
|-------|-------------------------------|-----|
| Pretrained parakeet-tdt-0.6b-v3 (greedy) | 16.53% | 4.29% |
| + Fine-tune 5 epochs (BN frozen, encoder unfrozen, lr=1e-6) | 14.06% | 2.90% |
| + Beam search (maes, beam=4) + token-level 4-gram LM (α=0.3) | **11.23%** | **2.61%** |

**Fine-tune:** -2.47pp WER, -1.39pp CER. CER drops harder — the
fine-tune mostly fixes morphological endings that the error analysis
flagged as 40% of baseline substitutions.

**Stack adds -2.83pp more WER.** Beam+LM alone on the pretrained
model didn't help much; the big LM win came AFTER fine-tuning, where
the acoustic model is more confident and the LM can break ties on
the remaining errors.

**Training cost:** 5 epochs × ~31 min on RTX 3090 = ~2.5h wall time
for ~19K LT speech clips. `numba-cuda==0.15.1` + WSL2 drivers, no
cloud GPU needed.

**LM build cost:** 211K LT Wikipedia articles → 2.6M sentences →
95M tokens. Tokenizing takes ~5 min, counting 4-grams ~3 min, writing
ARPA ~1 min. ~325 MB output file. ~32 GB RAM peak during counting.

## Cheat sheet for future Claude

1. **Freeze BN first.** Always. This is the #1 bug.
2. **Raw PyTorch loop.** Don't use Lightning for small fine-tunes.
3. **fp32.** No autocast. RNN-T is sensitive.
4. **Raw manifest text.** Don't lowercase/strip punctuation before training.
5. **Token-level LM.** Use the model's tokenizer + `chr(id + 100)`.
6. **Full test set for eval.** Not the first 200 clips.
7. **CER is informative.** Track it alongside WER.
8. **Smoke test checkpointing.** Before every full run.
9. **`numba-cuda==0.15.1` pin on WSL2.** Also `LD_LIBRARY_PATH=/usr/lib/wsl/lib:...`.
10. **Epoch 0 is often best.** Later epochs overfit. Save per-epoch and pick the best.
