# kurdish-scam-detector

Scam/phishing SMS detection for **Sorani Kurdish** and **Iraqi Arabic**, with a
per-message explanation of *why* a message was flagged.

CPU-only, trains in ~52 seconds, runs offline. Logistic regression over
character n-grams plus interpretable hand-built signals — chosen so the model
can show its reasoning, which a black box cannot.

---

## The problem

SMS and WhatsApp fraud is an everyday harm in Iraq and the Kurdistan Region.
Iraq ranks **8th of 193 countries** on the Global Organized Crime Index
(criminality score 7.13, 2023 assessment), 1st in Western Asia
([ocindex.net](https://ocindex.net/2023/country/iraq/)). **Zain Cash** is the
most-impersonated brand because it leads electronic payment in Iraq: between
22 January and 18 March, subscribers reported **783 fraud attempts that
failed**, and the company attributed those failures to public awareness
([964media](https://en.964media.com/16114/)) — awareness is exactly what a
detection tool scales. **Asiacell** has published a formal clarification
warning subscribers about SMS claiming they had won prizes and urging them to
call a number; it states the messages originate from services outside Iraq and
aim to drain subscriber credit
([asiacell.com](https://www.asiacell.com/en/about-us/news-and-event/press-releases/ClarificationtoallAsiacellsubscribers)).

The red flags recur across campaigns: unsolicited contact, generic greetings,
urgent deadlines, suspicious links, impersonation using free email domains
(`zain@hotmail.com` is not Zain), and requests for sensitive information.

The classifier covers eight archetypes: **prize/lottery**, **account
suspension**, **fake incoming transfer**, **OTP harvesting**, **fake job
offer**, **delivery/customs fee**, **charity fraud**, and **impersonated
support**.

## What is new here — and what is not

Arabic SMS spam and phishing detection is **already well published**. There is
prior work on AraBERT with dual feature extraction covering Modern Standard
Arabic *and Iraqi dialect*, hybrid CNN-LSTM models for Arabic and English SMS,
URL-based deep learning for Arabic smishing, and a public Arabic corpus of
68K+ messages. None of that is claimed here.

Kurdish Sorani has NLP resources too: [KLPT](https://github.com/sinaahmadi/klpt),
the Bochun stance dataset, the Adyan NER dataset, the Mabast semantic-
classification dataset, a Kurdish RoBERTa trained on ~188M tokens, and
[AS-RoBERTa](https://arxiv.org/abs/2507.18762) — script-specialised RoBERTa
models for Kurdish Sorani, Arabic, Persian and Urdu that beat mBERT/XLM-R by
2–5 points precisely because generic multilingual models mishandle
Arabic-script orthographic variation.

**What appears to be missing is a scam/phishing detection dataset or model for
Kurdish.** That claim was re-verified by web search in September 2026 across
searches for Kurdish/Sorani scam, smishing, fraud-SMS and spam datasets. The
Kurdish results that came back were stance detection (Bochun), NER (Adyan),
fake news (KurdFake), hate speech and text classification — no scam or phishing
corpus. The SMS-phishing results that came back (SmishTank, Mendeley SMS
phishing, the 153K SMS scam set) contain no Kurdish.

So, stated precisely: **no Kurdish scam-detection corpus or model that I could
find.** Not "the world's first" — absence of evidence in indexed sources is not
proof of absence, and unpublished or non-indexed work may well exist.

This reframes the project usefully:

- Arabic is the **well-resourced baseline** to compare against.
- Kurdish is the **low-resource target**.
- **Cross-lingual transfer Arabic → Kurdish** becomes a research question this
  repo can actually answer with an experiment. It does, below.

## Architecture

```
   raw message (Sorani Kurdish / Iraqi Arabic / mixed)
        │
        ├──────────────────────────────┐
        │                              │
        ▼                              ▼
  normalize.py                    signals.py
  ───────────────                 ─────────────────
  1 NFKC                          reads the RAW text, because it needs the
  2 strip zero-width / bidi         real domain that folding throws away
  3 strip tashkil + tatweel       ─────────────────
  4 fold ي ى ے→ی  ك→ک  ة→ه         · url present/count/shortener/
      أإآٱ→ا  ؤ→و                     suspicious TLD/raw-IP/brand-lookalike
      (ە and ئ kept: distinct        · phone present/count
       Kurdish letters)              · money present + magnitude bucket
  5 ٠-٩ ۰-۹ → 0-9                   · urgency / prize / sensitive-request /
  6 collapse letter runs 3+→2         action-link / pay-fee lexicons (ckb+ar+en)
      (digits exempt)                · asks_for_code  vs  provides_code
  7 lowercase, collapse space        · brand mention  ×  link / lookalike / urgency
        │                            · length, digit ratio, punctuation,
        ▼                              script-mixing ratio
  entity folding                      │
  __url__ __phone__                   │  28 features
  __money__ __num__                   │
        │                             │
        ▼                             │
  ┌─────────────────┐                 │
  │ char_wb 2-5 gram│                 │
  │ TF-IDF          │                 │
  ├─────────────────┤                 │
  │ word 1-2 gram   │                 │
  │ TF-IDF          │                 │
  └────────┬────────┘                 │
           │        FeatureUnion      │
           └───────────┬──────────────┘
                       ▼
           LogisticRegression(class_weight="balanced")
                       │
                       ▼
        label · calibrated probability · signed per-feature
        contributions (coef × value) · spans in the original text
                       │
                       ▼
        ┌──────────────────┴──────────────────┐
        ▼                                     ▼
  app.py (Streamlit)              export_web.py -> model.json
  desktop, RTL, ckb/ar/en         719 KB, 14,352 dims
                                              │
                                              ▼
                                  engine.js (same pipeline in JS)
                                  verified vs Python on all 3,321
                                  rows, max |dP| 2e-15
                                              │
                                              ▼
                                  demo.html - one file, opens in
                                  any browser, zero network calls
```

Entity folding applies to the n-gram model only. The *presence* of a link is
the signal; memorising one campaign's short URL is not learning, and on a
corpus this size it would quietly inflate the held-out score.

`normalize.normalize_traced` returns a character offset map alongside the
normalised string, so every highlight in the demo points at the characters the
user actually typed rather than at a post-normalisation approximation.

## Results

All numbers below are real output from `python src/evaluate.py` on this
repository. The full run is reproducible; nothing is copied from a paper.

**Read the caveat first: every row of the corpus is authored, not collected.
These measure pattern separability, not field performance.** See
[Known limitations](#known-limitations).

Splits are **group-aware**: all ~16 variants generated from one seed stay in
the same fold. Without that, template leakage returns a meaningless ~99%.

### Model comparison — 5-fold cross-validation, grouped on `seed_id`

Mean ± population std of scam-F1 across all five folds. One fold holds out only
~42 seeds, so single-fold numbers swing several points; these are the numbers
to quote.

| model | scam-F1 | ar | ckb |
|---|---|---|---|
| **Logistic regression (shipped)** | **0.930 ± 0.034** | 0.917 | 0.936 |
| Linear SVC | 0.921 ± 0.036 | 0.907 | 0.929 |
| Random forest | 0.896 ± 0.054 | 0.852 | 0.922 |
| Logistic regression, signals centred | 0.925 ± 0.033 | 0.932 | 0.916 |
| Logistic regression, no normalisation | 0.932 ± 0.027 | 0.928 | 0.927 |

Logistic regression wins outright, so the model that can explain itself is also
the most accurate one here — no trade-off had to be made. **No transformer is
used**, because the linear baseline has not been beaten: adding XLM-R or
Kurdish RoBERTa would mean a multi-GB dependency and a GPU assumption for no
measured gain on 208 seeds. That is a result, not an omission.

### Held-out detail (fold 0: 166 train seeds / 42 test seeds, 672 rows)

|  | precision | recall | F1 | support |
|---|---|---|---|---|
| ham | 0.869 | 0.949 | 0.908 | 336 |
| scam | 0.944 | 0.857 | 0.899 | 336 |

PR-AUC (average precision, scam): **0.982**

```
confusion matrix     pred_ham  pred_scam
  true_ham              319        17
  true_scam              48       288
```

Per language: `ar` P 0.883 / R 0.800 / F1 0.839 · `ckb` P 1.000 / R 0.909 / F1 0.952

Per-category recall on this fold: prize 1.00, phishing 1.00, otp 1.00,
transfer 1.00, delivery 0.75, job 0.50, support 0.50. **The weak cells are two
seeds wide** — `job` is n=32 rows from 2 seeds — so treat them as "needs more
seeds", not as a measured category weakness.

### Cross-lingual transfer

Grouped on `concept_id`, not `seed_id`. The two seed files are **parallel
translations** (`ar-scam-017` is `ckb-scam-017` in Arabic, down to the
amounts), so grouping on `seed_id` alone would put the Arabic twin of a Kurdish
test message into training and report it as transfer. It inflated `joint → ckb`
to 0.997 before this was caught. One concept split (21 held-out concepts) is
reused for every row, so the numbers are comparable.

| train → test | full model | n-grams only (no signals) |
|---|---|---|
| ar → ar (in-language) | 0.949 | 1.000 |
| ckb → ckb (in-language) | 0.949 | 0.917 |
| **ar → ckb (transfer)** | **0.886** | **0.834** |
| **ckb → ar (transfer)** | **0.952** | **0.724** |
| joint → ckb | 0.940 | — |
| joint → ar | 0.949 | — |

**Transfer works, and most of what crosses is not the script.** With the full
model, Arabic → Kurdish loses ~6 points against in-language (0.886 vs 0.949),
and Kurdish → Arabic loses nothing measurable. But the signals module is
language-independent by construction — its lexicons cover both languages
regardless of what was trained on — so the full-model columns credit hand-built
rules as "transfer". Disabling signals isolates the shared script: transfer
falls to 0.834 (ar→ckb) and 0.724 (ckb→ar), while in-language stays at
0.917–1.000. Character n-grams over a shared script carry real but **partial**
transfer; the cross-lingual signal layer is what closes the gap.

The `ar → ar` n-grams-only 1.000 is a 21-concept sample, not a claim of
perfection.

### Ablation (fold 0)

| configuration | scam-F1 | ckb |
|---|---|---|
| full (char + word + signals, normalised) | 0.899 | 0.952 |
| no normalisation | 0.917 | 0.952 |
| no entity folding | 0.906 | 0.952 |
| no normalisation, no folding | 0.930 | 0.952 |
| char n-grams only | 0.910 | 0.919 |
| word n-grams only | 0.903 | 0.919 |
| char + word, no signals | 0.919 | 0.928 |
| signals only | 0.903 | 0.907 |
| signals centred (StandardScaler) | 0.920 | 0.909 |

**Normalisation shows no in-distribution gain, and the design brief expected
one.** This was investigated rather than hidden. The reason is the evaluation
setup, not the normaliser: `build_dataset.py` perturbs train and test with the
same generator, so both halves contain the same spelling variants and the model
simply memorises both. Normalisation cannot help when nothing is unseen.

It earns its place under **orthographic shift**, which is the case that
actually matters — a scammer's spelling is not drawn from your training
distribution:

| | clean F1 | shifted F1 | delta |
|---|---|---|---|
| with normalisation | 0.899 | **0.906** | +0.007 |
| without normalisation | 0.917 | 0.871 | −0.047 |

Without normalisation the model loses 4.7 points when the test set is pushed
into unseen script variants. With it, performance holds. Normalisation buys
**robustness**, not in-distribution accuracy, and on this corpus those are
different things.

The signal scaler is `MaxAbsScaler`, not `StandardScaler`, primarily for
honesty of explanation: centering gives an *absent* feature a negative value,
which multiplied by a negative coefficient contributes *towards* scam — so the
demo would cite "company name plus a link" as evidence on a message containing
no link. Scaling without centering keeps absent signals at exactly zero. It
also happens to score marginally better in cross-validation (0.930 vs 0.925)
and clearly better on Kurdish (0.936 vs 0.916), so nothing was traded away.

### Adversarial probe

Evasions are applied to the **whole** held-out set, both classes, and none of
them appears in training.

Scoring only evaded *scam* messages is misleading and an earlier version of
this table did exactly that — it showed combined evasion reaching **1.000
recall**, which reads as the model getting *better* under attack. It is not.
Mangling text shreds it into rare n-grams that look nothing like clean ham, so
the model drifts toward "scam" for everything. The FPR column is what exposes
it.

| evasion | scam-F1 | scam recall | FPR on ham | F1 drop |
|---|---|---|---|---|
| baseline (no evasion) | 0.899 | 0.857 | 0.051 | — |
| digit-system swap | 0.899 | 0.857 | 0.051 | 0.000 |
| zero-width insertion | 0.887 | 0.878 | 0.101 | +0.011 |
| tatweel padding | 0.924 | 0.946 | 0.101 | −0.026 |
| combined (zw + tatweel + stretch) | 0.858 | 0.946 | **0.259** | +0.040 |
| letter stretching | 0.720 | 0.619 | 0.101 | +0.179 |
| **Latin transliteration** | **0.483** | **0.318** | 0.000 | **+0.416** |

Reading this honestly:

- **Digit-system swap costs the attacker nothing and gains nothing** — the
  normaliser folds `٠-٩` and `۰-۹` to ASCII, so the evasion is fully defeated.
  Zero drop.
- **Zero-width and tatweel are largely defeated too**, but note their FPR
  doubles (0.051 → 0.101). The normaliser strips them; the residual damage is
  that mangled text is unlike anything in training.
- **Combined evasion is the degenerate case.** Recall rises to 0.946 while FPR
  hits 0.259 — one in four legitimate messages flagged. A tool that flags
  everything is useless, and this row is why recall alone must never be
  reported.
- **Latin transliteration is the real failure.** F1 collapses to 0.483 and
  recall to 0.318 — the model misses two thirds of transliterated scams. The
  corpus contains no Latin-script Kurdish, so the character n-grams have
  nothing to match. This is the most exploitable weakness in the system and is
  the top item on the roadmap.

### Collected-data evaluation

```
No rows with source='collected' in the corpus.
```

There is no collected data yet, and the evaluator says so rather than blending
authored and collected rows. `python src/evaluate.py --collected-only` is
wired and will report separately as soon as real messages exist.

## Known limitations

**1. The entire corpus is synthetic.** All 3,321 rows are `source: "authored"`.
Not one is a real intercepted message. Every number above measures whether a
model can separate *authored scam patterns* from *authored legitimate
patterns*. It is not evidence of field performance, and it should not be
presented as such.

**2. The effective sample size is 208, not 3,321.** The corpus is 208
hand-written seeds expanded ~16× by a template expander. Group-aware splitting
means the model is genuinely tested on unseen seeds, but 42 held-out seeds per
fold is small — hence ±0.034 std across folds and per-category cells only two
seeds wide.

**3. Latin transliteration defeats it.** scam-F1 0.483, recall 0.318. A scammer
writing Sorani in Latin script evades this classifier today.

**4. Heavy combined obfuscation makes it cry wolf.** 0.253 false-positive rate
on legitimate messages under combined evasion — one legitimate message in four.

**5. The Kurdish seeds were drafted by an author who is not a native Sorani
speaker, then reviewed.** A native Sorani speaker has since read the seed files
and confirmed the wording. That closes the register risk on the 208 seeds, but
note what the review does and does not cover: it validates the seeds, not the
~16× template expansion built on top of them, whose perturbations are
mechanical and unreviewed.

**6. Normalisation gives no in-distribution gain** (§Ablation). It is justified
by robustness under orthographic shift, not by the headline number.

**7. KLPT is not used.** It would be the better citation for Kurdish
normalisation, but every release pins `cyhunspell`/`chunspell`, which publish
no wheels for Python 3.14 and need a hunspell toolchain to build.
`pip install klpt` fails with `ResolutionImpossible` on this interpreter, so
`src/normalize.py` implements the normalisation directly.

**8. Sorani only.** No Badini/Kurmanji. No voice notes, images, or
multi-message conversations.

**9. The lexicons are hand-built and will drift.** New campaigns use new
wording; nothing here adapts on its own.

**10. `ة → ه` folding is kept, while `ە` and `ئ` are preserved.** This means
Arabic `ه` and Kurdish `ە` remain distinct code points, which slightly limits
character n-gram overlap between the two languages. Preserving a real Kurdish
letter was judged more important than maximising transfer.

## Roadmap

1. **Collect real messages.** The single highest-value next step. The
   redaction protocol and contribution flow are in
   [`data/README.md`](data/README.md); `--collected-only` evaluation is already
   wired. Every headline claim stays provisional until this exists.
2. ~~Native-speaker review of the Kurdish seeds.~~ **Done** — a native Sorani
   speaker has reviewed and confirmed the seed wording. The remaining gap is
   reviewing a sample of the *expanded* rows, since the perturbations that
   generate them are mechanical.
3. **Close the Latin-transliteration hole.** Either add transliterated variants
   as a training augmentation, or transliterate-to-Arabic-script as a
   normalisation step so both forms map to one representation. The second is
   cleaner and reuses the existing offset-map machinery.
4. **Fine-tune Kurdish RoBERTa / AS-RoBERTa** and compare honestly against this
   baseline. The linear model is not beaten yet; if a transformer does not beat
   it on held-out seeds, that result gets reported too.
5. **Android SMS integration** with on-device inference. The shipped model is
   ~a few MB of sparse coefficients and runs on CPU in milliseconds, so
   on-device is realistic without a server — which also means no message ever
   leaves the phone.
6. **Per-category thresholds.** OTP-harvesting detection is near-perfect while
   job and delivery lag; one global threshold is leaving accuracy on the table.

## Setup

Requires **Python 3.11+** (developed and measured on 3.14, CPU only, no GPU).

```bash
git clone https://github.com/mahmoudyouns22/kurdish-scam-detector
cd kurdish-scam-detector
pip install -r requirements.txt
```

```bash
python src/build_dataset.py     # seeds -> data/corpus.csv   (~1s)
python src/train.py             # train + compare -> models/ (~52s)
python -m streamlit run src/app.py
```

Optional:

```bash
python src/evaluate.py                      # full experiment suite (~8 min)
python src/evaluate.py --collected-only     # real messages only, once they exist
python src/predict.py "پیرۆزە! ژمارەکەت 10000000 دینار بردویەتییەوە"
python src/normalize.py                     # normalisation demo
python src/signals.py                       # signal-extraction demo
```

Everything after `pip install` runs offline. The demo makes no network calls;
nothing you paste is uploaded or logged.

### CLI output

```
$ python src/predict.py "پیرۆزە! ژمارەکەت 10000000 دینار بردویەتییەوە. خێرا پەیوەندی بکە 0750 000 0000"

  Likely a SCAM   (100.0% scam)
  بە گومانەوە ساختەیە
  على الارجح احتيال

  why:
    scam +3.089  Contains a phone number
    scam +2.372  Message length
    scam +2.320  Size of the amount
    scam +1.371  Urgency wording
    scam +0.622  Prize or winning wording
```

## Repository layout

```
kurdish-scam-detector/
├── data/
│   ├── README.md          provenance, ethics, redaction + contribution flow
│   ├── seeds/*.jsonl      208 hand-authored seeds (tracked)
│   └── corpus.csv         generated, gitignored
├── src/
│   ├── normalize.py       Arabic-script normalisation + entity folding + offset map
│   ├── signals.py         28 interpretable features, with spans
│   ├── build_dataset.py   seeds -> labelled corpus
│   ├── train.py           train, compare 3 models, persist
│   ├── evaluate.py        held-out, CV, transfer, ablation, adversarial
│   ├── predict.py         inference + per-message explanation
│   └── app.py             Streamlit demo (RTL, ckb/ar/en)
├── models/                generated, gitignored
└── notebooks/             error analysis
```

## Scope and ethics

This is a **defensive** classifier and an evaluation harness. It is not, and
will not become, a scam generator, a bulk-SMS sender, or anything that helps
someone send fraudulent messages. `build_dataset.py` exists solely to produce
labelled training data: it has no send path, no recipient list and no delivery
integration.

The seed messages imitate scam *patterns* so the model can learn them. No real
person's data appears anywhere. Every phone number is `07XX 000 0000`; every
URL is invented, and none was resolved or visited.

## Author

**Mahmoud Youns Mahmoud**
Cybersecurity undergraduate, Tishk International University, Erbil
Co-founder & Head of Technology, [Nexaura](https://github.com/mahmoudyouns22)

GitHub: [@mahmoudyouns22](https://github.com/mahmoudyouns22) ·
Email: [mahmoudyouns2004110@gmail.com](mailto:mahmoudyouns2004110@gmail.com)

Related work: [`ai-cyber-defense-system`](https://github.com/mahmoudyouns22/ai-cyber-defense-system)
(Random Forest IDS over NSL-KDD with honeypot redirection and auto-blocking) ·
[`web-vuln-scanner`](https://github.com/mahmoudyouns22/web-vuln-scanner)

## License

MIT — see [LICENSE](LICENSE).
