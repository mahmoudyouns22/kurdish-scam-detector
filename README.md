# kurdish-scam-detector

Scam/phishing SMS detection for **Sorani Kurdish** and **Iraqi Arabic**, with a
per-message explanation of *why* a message was flagged.

CPU-only, trains in ~44 seconds, runs offline. Logistic regression over
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
  desktop, RTL, ckb/ar/en         1,268 KB, 26,158 dims
                                              │
                                              ▼
                                  engine.js (same pipeline in JS)
                                  verified vs Python on all 3,840
                                  rows, max |dP| 5.6e-15
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
~48 seeds, so single-fold numbers swing several points; these are the numbers
to quote.

| model | scam-F1 | ar | ckb |
|---|---|---|---|
| **Logistic regression (shipped)** | **0.961 ± 0.012** | 0.962 | 0.959 |
| Linear SVC | **0.966 ± 0.009** | 0.970 | 0.962 |
| Random forest | 0.908 ± 0.023 | 0.892 | 0.921 |
| Logistic regression, signals centred | 0.931 ± 0.018 | 0.941 | 0.920 |
| Logistic regression, no normalisation | 0.957 ± 0.009 | 0.962 | 0.951 |

**Linear SVC is the most accurate model, and it is not the one shipped.** On an
earlier corpus logistic regression led and there was no trade-off to make; with
the romanised, 240-seed corpus SVC edges ahead by 0.005 F1 — inside a fold
either way, but it is ahead. Logistic regression ships anyway, because it emits
calibrated probabilities and signed per-feature contributions, and the
explanation panel is the point of the tool. That is a deliberate ~0.005 paid for
interpretability, not a claim that the shipped model is the strongest.

**No transformer is used.** The linear baseline has not been beaten: adding
XLM-R or Kurdish RoBERTa would mean a multi-GB dependency and a GPU assumption
for no measured gain on 240 seeds. That is a result, not an omission.

### Held-out detail (fold 0: 192 train seeds / 48 test seeds, 768 rows)

|  | precision | recall | F1 | support |
|---|---|---|---|---|
| ham | 0.972 | 0.995 | 0.983 | 384 |
| scam | 0.995 | 0.971 | 0.983 | 384 |

PR-AUC (average precision, scam): **0.999**

```
confusion matrix     pred_ham  pred_scam
  true_ham              382         2
  true_scam              11       373
```

Per language: `ar` P 1.000 / R 0.948 / F1 0.973 · `ckb` P 0.990 / R 0.995 / F1 0.992

Per-category recall: prize 1.00, phishing 1.00, charity 1.00, delivery 1.00,
job 0.979, support 0.979, otp 0.917, **transfer 0.896**.

`job` and `delivery` were the two weak cells on the previous corpus — 0.500 and
0.750, from 6 seeds each. Four more seeds apiece took them to 0.979 and 1.000,
which is the expected result when a "weak category" was really a sample-size
artefact. `transfer` and `otp` are now the lowest, and the same caveat applies
to them: they are small cells, not established weaknesses.

### Cross-lingual transfer

Grouped on `concept_id`, not `seed_id`. The two seed files are **parallel
translations** (`ar-scam-017` is `ckb-scam-017` in Arabic, down to the
amounts), so grouping on `seed_id` alone would put the Arabic twin of a Kurdish
test message into training and report it as transfer. It inflated `joint → ckb`
to 0.997 before this was caught. One concept split (24 held-out concepts) is
reused for every row, so the numbers are comparable.

| train → test | full model | n-grams only (no signals) |
|---|---|---|
| ar → ar (in-language) | 0.928 | 0.958 |
| ckb → ckb (in-language) | 0.884 | 0.965 |
| **ar → ckb (transfer)** | **0.832** | **0.482** |
| **ckb → ar (transfer)** | **0.717** | **0.472** |
| joint → ckb | 0.859 | — |
| joint → ar | 0.883 | — |

**Transfer is real, and almost none of it is the script.** Arabic → Kurdish
loses ~10 points against in-language (0.832 vs 0.928); Kurdish → Arabic loses
more, to 0.717. Strip the signals and transfer collapses to ~0.48 both ways —
barely above chance on a balanced set — while in-language n-grams stay at
0.958–0.965. So character n-grams over a shared script transfer *poorly* on
their own, and the language-independent signal layer is doing nearly all the
cross-lingual work.

This is a sharper answer than the earlier corpus gave, and a less flattering
one. Before romanisation, n-grams-only transfer measured 0.834/0.724 and the
shared script looked like it carried much of the load. Widening the character
vocabulary to cover Latin appears to have specialised the n-grams further
toward each language's own surface forms. Whichever way it is read, the honest
version is: **do not credit the shared Arabic script for the transfer these
numbers show.**

### Ablation (fold 0)

| configuration | scam-F1 | ckb |
|---|---|---|
| full (char + word + signals, normalised) | 0.983 | 0.992 |
| no normalisation | 0.971 | 0.972 |
| no entity folding | 0.984 | 0.995 |
| no normalisation, no folding | 0.962 | 0.962 |
| char n-grams only | 0.948 | 0.918 |
| word n-grams only | 0.965 | 0.953 |
| char + word, no signals | 0.968 | 0.939 |
| signals only | 0.891 | 0.876 |
| signals centred (StandardScaler) | 0.948 | 0.923 |

**Normalisation now shows a small in-distribution gain — 0.983 with, 0.971
without — where on the earlier corpus it showed none at all.** The earlier null
result was an artefact of the evaluation setup rather than the normaliser:
`build_dataset.py` perturbs train and test with the same generator, so both
halves contain the same spelling variants and the model can memorise both.
Normalisation cannot help when nothing is unseen. Romanised rows widened the
orthographic spread enough for folding to start paying off directly.

Note also that **entity folding is not earning its place** here: removing it
scores 0.984 against 0.983. It is kept because memorising one campaign's short
URL is not learning and would inflate held-out scores on a corpus this size —
a methodological choice, not one these numbers support.

It earns its place under **orthographic shift**, which is the case that
actually matters — a scammer's spelling is not drawn from your training
distribution:

| | clean F1 | shifted F1 | delta |
|---|---|---|---|
| with normalisation | 0.983 | 0.863 | −0.120 |
| without normalisation | 0.971 | 0.847 | −0.124 |

On the earlier corpus this table was the whole justification for the normaliser:
without it the model lost 4.7 points under orthographic shift, with it
performance held. That gap has now almost closed — −0.120 against −0.124. Both
configurations degrade badly under shift, and normalisation barely separates
them.

Normalisation is still kept, and now mostly on the direct gain in the ablation
above (0.983 vs 0.971) rather than on robustness. The earlier, stronger claim
did not survive a larger and more orthographically varied corpus, and the
number that used to support it is left here rather than replaced.

The signal scaler is `MaxAbsScaler`, not `StandardScaler`, primarily for
honesty of explanation: centering gives an *absent* feature a negative value,
which multiplied by a negative coefficient contributes *towards* scam — so the
demo would cite "company name plus a link" as evidence on a message containing
no link. Scaling without centering keeps absent signals at exactly zero. It
also happens to score better in cross-validation (0.961 vs 0.931) and better on
Kurdish (0.959 vs 0.920), so nothing was traded away.

### Adversarial probe

Evasions are applied to the **whole** held-out set, both classes. None appears
in training, with one deliberate exception: Latin script is not really an
evasion, it is how a great deal of Kurdish is typed, and the classifier used to
fail on it outright. Training is now augmented with an ASCII-ish romanisation
(`sh`/`ch`/`3`), so the row below probes with a *different*, diacritic
Hawar-style one (`ş`/`ç`/`ê`) that training has never seen. Probing with the map
the model trained on would measure memorisation and report it as robustness.

Scoring only evaded *scam* messages is misleading and an earlier version of
this table did exactly that — it showed combined evasion reaching **1.000
recall**, which reads as the model getting *better* under attack. It is not.
Mangling text shreds it into rare n-grams that look nothing like clean ham, so
the model drifts toward "scam" for everything. The FPR column is what exposes
it.

| evasion | scam-F1 | scam recall | FPR on ham | F1 drop |
|---|---|---|---|---|
| baseline (no evasion) | 0.983 | 0.971 | 0.005 | — |
| digit-system swap | 0.983 | 0.971 | 0.005 | 0.000 |
| tatweel padding | 0.941 | 0.992 | 0.117 | +0.042 |
| zero-width insertion | 0.920 | 0.969 | 0.138 | +0.063 |
| letter stretching | 0.900 | 0.935 | 0.143 | +0.083 |
| **Latin transliteration** (unseen style) | **0.886** | **0.799** | **0.005** | +0.097 |
| combined (zw + tatweel + stretch) | 0.798 | 1.000 | **0.508** | +0.185 |

Reading this honestly:

- **Digit-system swap costs the attacker nothing and gains nothing** — the
  normaliser folds `٠-٩` and `۰-۹` to ASCII, so the evasion is fully defeated.
  Zero drop.
- **Latin transliteration went from the worst row to a middling one.** It was
  F1 0.483 / recall 0.318 before the training augmentation; it is now 0.886 /
  0.799 against a romanisation style the model has never seen. The FPR column
  is what makes this a real gain rather than a trade: it stays at 0.005, the
  same as baseline, so the extra recall is not bought by flagging legitimate
  messages. One scam in five still gets through romanised.
- **The other obfuscations got worse, and that is the cost of the fix.**
  Zero-width FPR went 0.101 → 0.138, tatweel 0.101 → 0.117, letter stretching
  0.101 → 0.143. Widening the character vocabulary to cover Latin appears to
  have made the model readier to fire on unfamiliar character sequences in
  general.
- **Combined evasion is the degenerate case and is now clearly worse.** Recall
  reaches 1.000 while FPR hits **0.508** — more than half of legitimate
  messages flagged, up from 0.259. That 1.000 recall is not the model winning;
  it is the model calling almost everything a scam once text is shredded into
  rare n-grams. This row is why recall must never be reported alone, and it is
  the clearest open regression in the project.

### Collected-data evaluation

```
No rows with source='collected' in the corpus.
```

There is no collected data yet, and the evaluator says so rather than blending
authored and collected rows. `python src/evaluate.py --collected-only` is
wired and will report separately as soon as real messages exist.

## Known limitations

**1. The entire corpus is synthetic.** All 3,840 rows are `source: "authored"`.
Not one is a real intercepted message. Every number above measures whether a
model can separate *authored scam patterns* from *authored legitimate
patterns*. It is not evidence of field performance, and it should not be
presented as such.

**2. The effective sample size is 240, not 3,840.** The corpus is 240
hand-written seeds expanded ~16× by a template expander. Group-aware splitting
means the model is genuinely tested on unseen seeds, but 48 held-out seeds per
fold is small — hence ±0.012 std across folds and per-category cells only a few
seeds wide.

**3. Latin transliteration is handled, not solved.** It used to defeat the
classifier outright (scam-F1 0.483, recall 0.318). Training is now augmented
with an ASCII-ish romanisation, and against an unseen diacritic Hawar-style
romanisation the model reaches scam-F1 0.886, recall 0.799 — with the
false-positive rate on legitimate messages *unchanged* at 0.005, so the gain is
not bought by flagging everything. One scam message in five still gets through
in Latin script, and only two romanisation styles have been tested.

**4. Heavy combined obfuscation makes it cry wolf, and the romanisation work
made this worse.** Under zero-width plus tatweel plus letter stretching, the
false-positive rate on legitimate messages is 0.508 — up from 0.253 before the
augmentation. Recall goes to 1.000 in that row, which looks good and is not:
the model is calling almost everything a scam once text is shredded into rare
n-grams. Widening the character vocabulary to cover Latin appears to have made
it readier to fire on unfamiliar character sequences generally. This is the
clearest open regression in the project.

**5. The Kurdish seeds were drafted by an author who is not a native Sorani
speaker, then reviewed.** A native Sorani speaker has since read the seed files
and confirmed the wording. That closes the register risk on the 240 seeds, but
note what the review does and does not cover: it validates the seeds, not the
~16× template expansion built on top of them, whose perturbations are
mechanical and unreviewed.

**6. Normalisation's benefit is small and was smaller before.** On the earlier
208-seed corpus the ablation showed no in-distribution gain at all, and
normalisation was justified purely by robustness under orthographic shift. With
the romanised corpus it now earns a modest direct gain (0.983 with, 0.971
without), and under orthographic shift the two are close (−0.120 vs −0.124).
The honest summary is that normalisation helps, less than the design intent
assumed.

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
3. ~~Close the Latin-transliteration hole.~~ **Done, partly.** Training is
   augmented with an ASCII-ish romanisation (`build_dataset._romanise`), and
   against an unseen Hawar-style romanisation scam-F1 went 0.483 → 0.886. Two
   things remain: one scam in five still gets through romanised, and the
   augmentation regressed the combined-obfuscation FPR from 0.259 to 0.508.
   Transliterate-to-Arabic-script *as a normalisation step* is still the
   cleaner fix — it would collapse both orthographies into one representation
   instead of asking the model to learn both, and would not widen the character
   vocabulary the way augmentation did.
4. **Fine-tune Kurdish RoBERTa / AS-RoBERTa** and compare honestly against this
   baseline. The linear model is not beaten yet; if a transformer does not beat
   it on held-out seeds, that result gets reported too.
5. **Android SMS integration** with on-device inference. The shipped model is
   ~a few MB of sparse coefficients and runs on CPU in milliseconds, so
   on-device is realistic without a server — which also means no message ever
   leaves the phone.
6. **Per-category thresholds.** `transfer` (0.896) and `otp` (0.917) now trail
   the categories that sit at 1.000; one global threshold is leaving accuracy on
   the table.
7. **Shrink the browser bundle.** Adding romanised text grew the exported
   vocabulary and took `model.json` from 719 KB to 1,268 KB, and `demo.html`
   from 768 KB to 1,317 KB. For a tool aimed at people on Iraqi mobile data
   that is a real cost, and pruning low-weight n-gram features should recover
   most of it.

## Setup

Requires **Python 3.11+** (developed and measured on 3.14, CPU only, no GPU).

```bash
git clone https://github.com/mahmoudyouns22/kurdish-scam-detector
cd kurdish-scam-detector
pip install -r requirements.txt
```

```bash
python src/build_dataset.py     # seeds -> data/corpus.csv   (~1s)
python src/train.py             # train + compare -> models/ (~44s)
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
│   ├── seeds/*.jsonl      240 hand-authored seeds (tracked)
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
