# Data: provenance, ethics, and how to contribute

## What is in here

| Path | Tracked | What it is |
|---|---|---|
| `seeds/scam_ckb.jsonl` | yes | 52 Sorani Kurdish scam messages, hand-authored |
| `seeds/ham_ckb.jsonl` | yes | 52 Sorani Kurdish legitimate messages, hand-authored |
| `seeds/scam_ar.jsonl` | yes | 52 Iraqi Arabic scam messages, hand-authored |
| `seeds/ham_ar.jsonl` | yes | 52 Iraqi Arabic legitimate messages, hand-authored |
| `corpus.csv` | **no** (gitignored) | 3,321 rows expanded from the seeds |

The seeds are the source of truth. `corpus.csv` is disposable — rebuild it with
`python src/build_dataset.py`.

## Provenance — read this before quoting any number

**Every row currently in this repository is `source: "authored"`.** Not one is a
real intercepted message. The seeds were written by hand to imitate the scam
*patterns* documented in the sources cited in the top-level README (Zain Cash
impersonation campaigns, the Asiacell prize-SMS clarification, and the recurring
red flags those reports describe). They were not copied from real messages.

This matters for how results should be read: metrics on this corpus measure
**pattern separability** — can a model tell these authored scam patterns from
these authored legitimate patterns — and **not field performance**. A real
inbox contains message types nobody thought to author, and real scammers adapt
faster than a seed file does.

`src/evaluate.py` reports the authored and collected splits separately and
refuses to blend them. Run `python src/evaluate.py --collected-only` once real
data exists.

### Record format

```json
{"id": "ckb-scam-017", "text": "...", "label": "scam", "lang": "ckb",
 "category": "transfer", "source": "authored"}
```

- `label` — `scam` or `ham`
- `lang` — `ckb` (Sorani Kurdish) or `ar` (Iraqi Arabic)
- `category` — `prize`, `phishing`, `transfer`, `otp`, `job`, `delivery`,
  `charity`, `support`, or `legit`
- `source` — **`authored`** (written for this dataset) or **`collected`**
  (a real message, redacted). Never mix the two silently.

`build_dataset.py` also derives a `concept_id` by stripping the language
prefix. The two seed files are parallel translations — `ar-scam-017` is
`ckb-scam-017` in Arabic — so the cross-lingual experiments group on
`concept_id` to stop a message's own translation sitting in the training set.

## A note on the placeholders

Every phone number is of the form `07XX 000 0000`. Every URL is invented — no
observed attacker infrastructure is reproduced here, and none of these hosts
was resolved or visited. They are shaped to look real so the URL and phone
signals have something to fire on; they are not real.

## Ethics and scope

This is a **defensive** dataset for a **defensive** classifier.

- `build_dataset.py` is a training-data generator. It has no send path, no
  recipient list and no delivery integration, and it will not be given one.
- Do not use these seeds as templates for sending anything to anyone.
- No real person's data appears here: no real names, real numbers, real account
  identifiers, or real victims' messages.

## Contributing real messages

Real collected data is the single most valuable thing this project is missing.
If you want to contribute:

**1. Redact before the message leaves your device.** Replace, in this order:

| Replace | With |
|---|---|
| Any phone number | `0750 000 0000` |
| Personal names | `[NAME]` |
| Account / card / IBAN / national ID numbers | `[ACCOUNT]` |
| OTP codes and PINs | a random 6-digit number |
| The scammer's URL host | keep the *shape*, change the host (`badhost.top/xy`) |
| Addresses, workplaces, anything identifying | `[REDACTED]` |

Keep the wording, the urgency, the misspellings and the punctuation. Those are
the signal. Do **not** clean up the grammar.

**2. Add one JSON object per line**, to `seeds/scam_ckb.jsonl`,
`seeds/ham_ckb.jsonl`, `seeds/scam_ar.jsonl` or `seeds/ham_ar.jsonl`:

```json
{"id": "ckb-scam-101", "text": "...", "label": "scam", "lang": "ckb",
 "category": "prize", "source": "collected"}
```

`id` must be unique. `source` must be `"collected"`.

**3. Include legitimate messages too.** They are harder to get and worth more.
A promotional SMS from Asiacell with urgency, a brand, a link and a price is the
adversarial case for this classifier — it looks exactly like a scam and is not
one. A corpus of only scams teaches a model to cry wolf.

**4. Rebuild and check:**

```bash
python src/build_dataset.py
python src/evaluate.py --collected-only
```

**5. Do not contribute** a message that identifies a victim, anything from a
private conversation you were not party to, or anything you are not free to
share.

## Known gaps in the current seeds

- Drafted by one author. The Kurdish seeds especially need review by more
  native speakers — register and naturalness matter more here than volume,
  because every seed is amplified ~16× by the expander.
- Sorani only. No Badini/Kurmanji, no Latin-script Kurdish as *input* (it
  appears only as an adversarial transform in `evaluate.py`).
- No voice notes, no images, no multi-message conversations — single SMS-shaped
  strings only.
- Scam archetypes are those documented in public reporting as of early 2026.
  New campaigns will not be represented until someone adds them.
