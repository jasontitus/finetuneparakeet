# Lithuanian LM corpus sources

Sources to add to `sliderforthewin/lt-asr-lm-corpora` for a
comprehensive Lithuanian language model.

## Currently in the repo

| source | sentences | register | notes |
|--------|-----------|----------|-------|
| CC-100 | 56.9M | web crawl (noisy) | Hurts beam+LM WER on formal speech — too informal |
| Wikipedia LT | 2.9M | encyclopedic | Clean, formal. Core of the best-performing LM |
| OpenSubtitles | 1.3M | informal dialogue | Minor contributor |

## Best-performing LM so far

**Domain 5-gram** (Wikipedia + training manifests only): **9.51% WER**
on CV25 LT. The smaller, cleaner corpus beat the 61M-sentence broad
corpus. Domain match > corpus size for LM fusion.

## Sources to add

### Parliamentary / formal speech (fixes VoxPopuli regression)

1. **Europarl Lithuanian** (~600k sentences)
   - European Parliament proceedings translated to Lithuanian
   - HF: `europarl_bilingual` (lang1=en, lang2=lt) — extract the LT side
   - Or: OPUS download at `opus.nlpl.eu/Europarl.php`
   - Register: formal parliamentary, exactly what VoxPopuli tests

2. **EUR-Lex Lithuanian** (~2-5M sentences)
   - EU legislation and legal texts in all EU languages
   - OPUS: `opus.nlpl.eu/EUbookshop.php` or `opus.nlpl.eu/JRC-Acquis.php`
   - Register: formal legal. Helps with parliamentary vocabulary

3. **Lithuanian Seimas transcripts**
   - National parliament session transcripts
   - Source: `lrs.lt/sip/portal.show?p_r=35403` (official stenograms)
   - Needs scraping/download script — no bulk download API
   - Register: exactly Lithuanian parliamentary speech
   - **NOT in ParlaMint** (CLARIN ParlaMint 4.0 includes LV/Latvia
     but not LT/Lithuania — verified 2026-04-12)
   - Would be unique and high-value. Sessions go back to 1990.
   - TODO: write a scraper for lrs.lt stenograms

4. **Europarl Lithuanian** (~634k sentences) ✓ DOWNLOADED
   - EU Parliament proceedings (not Lithuanian Seimas)
   - HF: `europarl_bilingual` config `en-lt`, extract LT side
   - Already saved at `data/lm/europarl_lt.txt` (92 MB)
   - Formal parliamentary register, helps VoxPopuli domain

### News / broadcast (general quality improvement)

4. **Leipzig Corpora Collection — Lithuanian**
   - `wortschatz.uni-leipzig.de/en/download/Lithuanian`
   - News articles, web text. 1M-10M sentences available
   - Register: news (clean, formal)

5. **mC4 Lithuanian** (subset of C4 multilingual)
   - HF: `mc4` config `lt`
   - Web crawl but cleaner than CC-100 (Google's quality filtering)
   - Very large (potentially 100M+ sentences)
   - Risk: same domain-mismatch as CC-100

### Books / literature

6. **Lithuanian LibriVox text** (if aligned transcripts exist)
   - Would be ideal for ASR LM since the audio domain matches
   - Likely very small

## Recommended build strategy

**Tiered interpolation** rather than one monolithic corpus:

```
P(w) = λ₁ × P_manifests(w)     # training domain (highest weight)
     + λ₂ × P_wikipedia(w)     # clean encyclopedic
     + λ₃ × P_europarl(w)      # parliamentary
     + λ₄ × P_eurlex(w)        # formal legal
     + λ₅ × P_subtitles(w)     # informal
     + λ₆ × P_cc100(w)         # broad web (lowest weight)
```

Tune λ weights on the dev sets for each target benchmark:
- CV25 → high λ₁, λ₂
- VoxPopuli → high λ₃, λ₄
- FLEURS → balanced

KenLM's `lmplz` can build per-domain LMs, then SRILM's `ngram
-mix-lm` or a simple Python script interpolates them.

## Adding to the HF repo

```bash
# Example: download Europarl LT, extract LT text, compress, upload
python -c "
from datasets import load_dataset
ds = load_dataset('europarl_bilingual', lang1='en', lang2='lt', split='train')
with open('europarl.sentences.txt', 'w') as f:
    for s in ds:
        lt = s['translation']['lt']
        if lt.strip():
            f.write(lt.strip() + '\n')
"
zstd europarl.sentences.txt
huggingface-cli upload sliderforthewin/lt-asr-lm-corpora \
    europarl.sentences.txt.zst corpora/europarl.sentences.txt.zst \
    --repo-type dataset
```
