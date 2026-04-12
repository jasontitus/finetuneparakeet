# VoxPopuli Lithuanian regression analysis

## The problem

Beam+LM decoding improves CV25 LT dramatically (14.06% → 9.40% WER)
but **makes VoxPopuli LT worse** (29.76% → 39.27% WER). This is
counterintuitive — the LM is supposed to help.

## Root cause: beam collapse on out-of-domain vocabulary

8 of 42 VoxPopuli test clips (19%) produce **completely empty output**
when using beam+LM decoding. Each empty clip scores 100% WER, dragging
the aggregate from ~30% to ~39%.

The beam collapse mechanism:
1. NeMo's MAES beam decoder scores candidates as
   `acoustic_score + α × LM_score`
2. When the LM has never seen a token sequence (e.g., parliamentary
   terms like `apsisprendimo teisę`, `konfrontuojančias šalis`,
   `ESBO ODIHR rekomendacijos`), it assigns a very negative log
   probability
3. With α=0.5, the LM penalty overwhelms the acoustic score
4. ALL beam candidates score below the decoder's internal pruning
   threshold
5. Every beam dies → empty output

## Evidence

```
Empty beam+LM outputs: 8 / 42 clips

REF: tuo tarpu europos sąjunga nesustoja klydusi santykiuose su baltarusija.
HYP: (empty)

REF: esbo odihr rekomendacijos absoliučiai ignoruojamos toliau vykdomi mirties nuosprendžiai...
HYP: (empty)

REF: pripažindami tautų apsisprendimo teisę privalome pasiekti kad dėl valstybės narystės...
HYP: (empty)
```

The greedy decoder handles these clips at ~30% WER (imperfect but
functional). The beam+LM decoder returns nothing.

Additional non-empty regressions: the LM "corrects" correct
parliamentary words to Wikipedia-common alternatives:
- `nord stream` → `northstream`
- `valstybėms` → `valstybė`

## Per-clip statistics

| metric | count |
|--------|-------|
| Total clips | 42 |
| Perfect (0% WER) | 2 |
| >50% WER | 12 |
| 100% WER (empty output) | 8 |

## Why the LM helps on CV25 but hurts on VoxPopuli

The LM (europarl+wiki+subs 5-gram) was trained on:
- Training manifest transcripts (CV25/FLEURS/shunyalabs) — crowdsourced
  read speech
- Lithuanian Wikipedia — encyclopedic, formal but not parliamentary
- Europarl — EU Parliament (different vocabulary from Lithuanian
  national parliament)
- OpenSubtitles — informal dialogue

VoxPopuli LT is Lithuanian national parliament (Seimas) speech. The
vocabulary includes:
- Lithuanian legal/constitutional terms
- Seimas-specific procedural language
- Proper nouns (ESBO, ODIHR, Nord Stream)
- Formal Lithuanian morphological forms rarely used in Wikipedia

The LM penalizes these tokens because they're out-of-vocabulary or
extremely rare in its training data.

## Legitimate fixes

### 1. Better LM coverage (the right fix)

Add Lithuanian Seimas transcripts to the LM training corpus. Sources:
- **lrs.lt** — official Seimas stenograms (1990-present). Needs
  a scraper; no bulk download API.
- **ParlaMint-LT** — NOT available (ParlaMint 4.0 includes Latvia
  but not Lithuania, verified 2026-04-12).

This would give the LM vocabulary coverage for exactly the terms
that cause beam collapse. See `LM_SOURCES.md` for details.

### 2. Graceful beam fallback (algorithmic fix)

Detect when all beams die and automatically retry with α=0 (pure
acoustic). This degrades gracefully to greedy quality instead of
producing empty output. Not currently implemented in NeMo's MAES
decoder.

### 3. Per-domain alpha tuning (pragmatic but not ideal)

Use α=0.5 for CV25/FLEURS (in-domain) and α=0.1 for VoxPopuli
(out-of-domain). This reduces the LM's influence on unfamiliar
vocabulary. Works but requires knowing the domain in advance.

## What to report

Beam+LM is not a universal improvement — it's a domain-specific
boost. Benchmarks should show both configurations:

| config | CV25 | FLEURS | VoxPopuli |
|--------|------|--------|-----------|
| Greedy | 14.06% | 19.64% | **29.76%** |
| Beam+LM (europarl 5-gram) | **9.40%** | **15.22%** | 39.27% |

The honest conclusion: beam+LM gives 33-38% relative improvement
on read speech / broadcast data, but regresses on out-of-domain
parliamentary speech due to LM vocabulary gaps. Fixing the LM
(adding Seimas data) is the path to universal improvement.

## Next steps

1. Build a Seimas transcript scraper for lrs.lt
2. Add Seimas text to the LM corpus
   (`sliderforthewin/lt-asr-lm-corpora`)
3. Rebuild the 5-gram LM with parliamentary coverage
4. Retest VoxPopuli — expect beam collapse to resolve
5. Consider implementing greedy fallback for empty beam outputs
   as a safety net regardless
