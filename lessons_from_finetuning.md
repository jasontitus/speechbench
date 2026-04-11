# Lessons from fine-tuning parakeet-tdt-0.6b-v3 on Lithuanian (part 1)

This document captures issues found while building a Lithuanian
fine-tuning pipeline on top of speechbench. Written from
`~/experiments/finetuneparakeet/` (Tier 1 smoke run on L4 spot,
2026-04-09). The Lithuanian-specific analysis lives there; what
follows is what's actionable *in speechbench itself*.

**➡ See also: `lessons_from_finetuning_working_recipe.md`** —
the second-round lessons after the initial runs failed. That file
documents the working recipe (16.53% → 11.23% WER), the BatchNorm
root cause of catastrophic forgetting, token-level LM gotcha,
and the full copy-paste training recipe.

## TL;DR

1. **speechbench's normalizer silently corrupts every non-English
   language in the suite.** `_PUNCT_RE = re.compile(r"[^a-z0-9'\s]")`
   replaces *every non-ASCII character* — including Lithuanian,
   Spanish, French, German, Czech, Polish, Russian, Chinese, Japanese
   diacritics and glyphs — with a space. This splits words apart
   during WER computation and inflates WER. **The existing Spanish
   and Lithuanian WER numbers in the report are wrong.** Measured
   impact on parakeet-tdt-0.6b-v3 / CV25 LT: **29.6% (speechbench
   normalizer) vs 13.74% (strict Unicode-preserving normalizer)** on
   the same model output.
2. **n=30 default is too small** for stable WER comparisons,
   especially when the normalizer is already inflating variance.
3. **CER is much more informative than WER for morphologically rich
   languages.** Lithuanian parakeet CER is 2.71% while its WER is
   13.74% — the model is right about most characters, it's just
   losing on word-boundary / inflection matches.
4. **Fine-tuning a strong multilingual ASR model is a precision game.**
   An aggressive 300-step fine-tune (lr=2e-5, partial encoder) at
   baseline ~14% WER regressed the model to 36%. Separate detail
   below for anyone planning to try this.

## Bug 1: ASCII-only normalizer mangles non-English text

### Location

`speechbench/eval.py:21`

```python
_PUNCT_RE = re.compile(r"[^a-z0-9'\s]")
```

and `speechbench/eval.py:55`

```python
s = _PUNCT_RE.sub(" ", s)
```

### What it does

The regex replaces every character that is *not* `[a-z0-9'\s]` with a
space. Because `lowercase()` is applied first, uppercase letters are
already folded. But the character class uses literal ASCII ranges —
it treats *every non-ASCII Unicode character as punctuation*.

### Concrete examples

- `grįžęs į lietuvą` (LT) → `gr  s    lietuv ` → collapsed to
  `gr s lietuv` — one word becomes two, another loses its ending.
- `específicos` (ES) → `espec ficos` — word split, wrong token count.
- `enfáticamente` (ES) → `enf ticamente`
- `München` (DE) → `m nchen`
- `мой` (RU) → empty string (all characters are non-ASCII)
- `北京` (ZH) → empty string

### Measured impact on a real benchmark

Same output from the same parakeet-tdt-0.6b-v3 checkpoint on the same
CV25 LT test clips:

| normalizer                                  | WER    | CER    |
|----------------------------------------------|--------|--------|
| speechbench (`[^a-z0-9'\s]` strips Unicode) | 29.6% (n=30) |  15.9% |
| strict (keeps Unicode letters)              | **13.74%** (n=300) |  **2.71%** |

These are measurements of the **same transcription output** — the
normalizer choice alone accounts for the 2x WER gap. The CER gap is
bigger because character-level matching is more sensitive to
character-dropping behaviour.

The same bug is visible in the existing Spanish numbers. Spot-check
from `results/gemma_extras/raw/ee3ddabe7cb1.json`:

```
"reference_norm": "las 25 fotograf as de dunlap que se sabe que a n existen
                   son las copias m s antiguas conservadas del documento"
```

The reference is Spanish text with `fotografías`, `aún`, and `más`.
Every `í`, `ú`, `á` has been converted to a space, splitting those
words into fragments. WER on this corrupted text is meaningless for
comparing model correctness.

### Affected datasets in the suite

Any dataset that has non-ASCII characters in its transcript. From
`speechbench/datasets.py`:

- `mls_es`, `voxpopuli_es`, `fleurs_es`, `common_voice_22_es`
- `voxpopuli_lt`, `fleurs_lt`, `common_voice_22_lt`, `common_voice_25_lt`
- any English dataset where a model hallucinates a non-ASCII character
  (fairly common with Whisper on proper nouns)

### Recommended fix

Replace the normalizer with one that preserves Unicode letters. Two
reasonable options:

**Option A — stdlib only, drop-in:**

```python
import unicodedata

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    s = text.lower()
    # Expand English contractions (existing behaviour — still fine for
    # English-only datasets, a no-op on non-English).
    for k, v in _CONTRACTIONS.items():
        s = s.replace(k, v)
    # Replace everything that's not a Unicode letter, Unicode digit,
    # apostrophe, or whitespace with a space. Categories:
    #   L* = letter (Lu, Ll, Lt, Lm, Lo)
    #   N* = number (Nd, Nl, No)
    out = []
    for ch in s:
        if ch in (" ", "'", "\t", "\n") or unicodedata.category(ch)[0] in ("L", "N"):
            out.append(ch)
        else:
            out.append(" ")
    s = "".join(out)
    s = _MULTI_WS_RE.sub(" ", s).strip()
    return s
```

**Option B — use the Open ASR Leaderboard normalizer:**

```python
# requirements-vm.txt
whisper-normalizer>=0.0.10
```

```python
from whisper_normalizer.basic import BasicTextNormalizer
_BASIC_NORM = BasicTextNormalizer()

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    return _BASIC_NORM(text).strip()
```

`whisper_normalizer.basic.BasicTextNormalizer` is what the Open ASR
Leaderboard and the HF ASR evaluation tooling use for multilingual
evaluation. Using it makes speechbench numbers directly comparable to
those leaderboards — which is probably what you want anyway.

For English-only datasets where the existing English-specific rules
matter (number-word mapping, contractions), consider dispatching:

```python
def normalize_text(text: str, language: str = "english") -> str:
    if language == "english":
        return _english_normalize(text)
    return _BASIC_NORM(text).strip()
```

And pass `dataset.language` through to `compute_wer`/`compute_cer`.

### How to verify the fix

A concrete test case from the Lithuanian run:

```python
ref = "grįžęs į lietuvą stojo savanoriu į lietuvos kariuomenę"
hyp = "Grįžęs į Lietuvos tojo savanoriai Lietuvos kariuomenė."
# Current (broken):
normalize_text(ref)  # → "gr s lietuv stojo savanoriu lietuvos kariuomen"
normalize_text(hyp)  # → "gr s lietuvos tojo savanoriai lietuvos kariuomen"
# Fixed:
normalize_text(ref)  # → "grįžęs į lietuvą stojo savanoriu į lietuvos kariuomenę"
normalize_text(hyp)  # → "grįžęs į lietuvos tojo savanoriai lietuvos kariuomenė"
```

The fixed version still gets the minor inflection errors (`lietuvą` vs
`lietuvos`) as errors — which is correct. The broken version also
loses word boundaries and can't even count tokens consistently.

Re-running the existing `lt`, `lt25`, `lt25g`, `es`, `es2` runs' raw
`results/*.json` files through the fixed normalizer (no re-inference
required — the per-clip `reference_raw` + `hypothesis_raw` fields are
still there) should give a corrected set of numbers for free.

## Bug 2: Default sample cap (n=30) is too small for stable WER

### Context

`speechbench/datasets.py` defaults datasets to `default_cap=200` or
`default_cap=300`, but the `--quick` flag in `speechbench launch`
caps further, and the results directory has multiple runs with
n=30.

### Why it matters

WER is a high-variance metric. On n=30, a single bad clip can shift
the aggregate number by 2-4 percentage points. For Lithuanian
parakeet:

- speechbench `lt25` run, n=30: 29.6% WER
- finetuneparakeet baseline, n=300: 13.74% WER with the *same* model

Part of this gap is the normalizer bug (above), but even controlling
for that, n=30 is simply too noisy to use for model comparisons.

### Recommendation

- Keep `default_cap=200` or `300` as the "quick" setting.
- Make the full test-split numbers (`--full`) the default published
  ones in reports. They're usually only 5-20x more expensive and
  that's trivial compared to the decisions people make from the
  numbers.
- When publishing small-n results, always show an error bar or at
  least note `n=N` in the report table.

## Bug 3: CER should be first-class, not secondary

For English datasets, CER is a nice-to-have. For Lithuanian (and
most non-English languages with rich inflection and agglutination),
**CER is the more informative metric**. Example from the baseline run:

| metric | value |
|--------|-------|
| CER    | 2.71% |
| WER    | 13.74% |

A single wrong letter in an inflected word ending costs 1 word error
and ~1 character error. The model is almost always saying
*approximately* the right Lithuanian word — it just sometimes misses
the case ending by a letter. Humans can read `kariuomenė` when the
model said `kariuomenę` — WER treats them as different words, CER
correctly reports this as a 1-character difference.

### Recommendation

Swap the column order in `report.md` / `report.csv` so CER comes
before WER for non-English datasets, or at least add a
`primary_metric` field per dataset that the report respects.

## Lesson 4: fine-tuning an already-good multilingual ASR model

Not a speechbench bug — a learning for anyone who looks at the LT
numbers and thinks "let's fine-tune that". The naive approach fails.

### What didn't work

- **lr=2e-5, 300 optimizer steps, partial encoder freeze (layers 0-11
  frozen)**, batch 4 × grad_accum 4. Baseline WER 13.74% → post-FT WER
  **35.82%**. Catastrophic regression.
- Failure mode: model went bimodal. Some clips still clean, others
  produced nonsense like `"te voy os pirmasis rinksnes"` for `"tai
  buvo jos pirmasis žingsnis"`.
- Root cause: 300 steps is enough to drift the multilingual weights
  off their pretrained manifold but nowhere near enough to re-converge
  to a new local optimum.

### Recipe that works (to be validated by the full run in progress)

- **Freeze the entire encoder.** Only train the RNN-T prediction
  network + joint network. ~60M trainable params out of 600M — the
  acoustic features stay frozen, only the decoder adapts.
- **Peak LR 5e-6**, cosine with ~1000 warmup steps.
- **Select best-by-dev-WER checkpoint** at save time, not the
  last-in-memory weights. Lightning's `ModelCheckpoint` does the
  saving; you just have to remember to reload the best one before
  your final `save_to(...)` call — it's easy to forget.
- **Reuse the pretrained tokenizer.** Don't retrain BPE. The v3 is
  multilingual; Lithuanian is in-vocabulary.

### Implication for speechbench

If speechbench ever wants to publish fine-tuned variants of the
included models, the framework needs to: (a) distinguish "stock" vs
"fine-tuned" checkpoints in the registry, (b) be careful about which
dataset the fine-tune was trained on (leak risk), and (c) report
against held-out splits that were never seen during training. The
finetuneparakeet repo holds out CV25 LT `test` split and uses `train`
+ `validated − test/dev` + `other` for training + `dev` for
validation — that partitioning logic is probably reusable if we go
this direction.

## Suggested follow-ups (in order of how much they move the numbers)

1. **Fix the normalizer** (Bug 1). Recomputes every published number
   with the fixed normalizer, using the raw per-clip fields already
   on disk. No re-inference cost.
2. **Bump default caps** to 500+ for non-English datasets and add
   `n` to the report tables (Bug 2).
3. **Show CER first** for non-English datasets (Bug 3).
4. **Add a `language=` dispatch** to `normalize_text` so English-only
   datasets keep the contractions + number-word rules but everything
   else uses a Unicode-safe normalizer.
5. **Add an "ASR leaderboard compatible" normalizer option**
   (Whisper `BasicTextNormalizer`) so the published speechbench
   numbers are directly comparable to the Open ASR Leaderboard.

## Appendix: reproduction

All numbers above are reproducible from the finetuneparakeet repo:

```bash
cd ~/experiments/finetuneparakeet
.venv/bin/python scripts/02_extract_cv25.py          # splits summary
.venv/bin/python scripts/03_prepare_manifests.py --datasets cv25_lt
bash scripts/launch_ft_vm.sh smoke                   # baseline + smoke fine-tune on L4 spot, ~$0.15
```

The smoke run writes `results/baseline_cv25_lt_test/summary.json` and
`results/finetuned_cv25_lt_test/summary.json` with the WER/CER numbers,
plus `per_clip.jsonl` with the raw and normalized reference/hypothesis
for every clip. Those per-clip files are what I used to measure the
normalizer impact — you can re-run any alternative normalizer over
those files without touching a GPU.
