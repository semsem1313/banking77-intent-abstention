# Project journal and viva notes

A step-by-step record of what was built, **why**, and **where it is in the code**. Each entry ends with likely viva questions and short answers.
New entries are added every time something changes. The newest entry is at the bottom.

**How to read the code links:** `src/split.py:74` means file `src/split.py`, line 74.

---

## Contents

0. [The big picture](#0-the-big-picture)
1. [Pre-registration](#1-pre-registration-oct-29)
2. [Getting the data reproducibly](#2-getting-the-data-reproducibly-oct-3)
3. [Repository hygiene](#3-repository-hygiene-oct-29)
4. [Packages](#4-packages-oct-30)
5. [Text normalization and features](#5-text-normalization-and-features-oct-30)
6. [The split](#6-the-split-oct-30)
7. [Amendment: near-duplicate sensitivity check](#7-amendment-near-duplicate-sensitivity-check-oct-29)
8. [First findings from the split](#8-first-findings-from-the-split-oct-30)
9. [Leakage guard and tests](#9-leakage-guard-and-tests-oct-30)

---

## 0. The big picture

**The question.** Why do classical intent classifiers fail on banking customer messages, and when should they abstain (hand off to a human) instead of answering?

**The pipeline**

```
download_data → split → baselines / tune (OOF predictions) → analysis (H1–H4, U) → evaluate_final (test, once) → export_web → website
```

**Two principles drive every decision**

1. **The test set is touched once.** Every choice (hyperparameters, thresholds, hypothesis verdicts) is made using *out-of-fold* (OOF) predictions on the training data.
   - OOF means each training message is predicted by a model that was **not** trained on it. This happens in 5-fold cross-validation: each fold is predicted by a model trained on the other 4.
   - That gives an honest prediction for every training message without using the test set.
2. **Rigor over accuracy.** The rubric gives 80 of 100 points to reasoning. A modest result with a sound evaluation beats a high score that hides leakage.

---

## 1. Pre-registration (Oct 29)

**What.** Before any model was trained, [PREREGISTRATION.md](PREREGISTRATION.md) was written and committed. It fixes:
- the four hypotheses (H1–H4) and the unseen-intent question (U);
- **what result counts as support** for each;
- the split rules;
- every constant (seeds, thresholds, number of bootstrap resamples).

**Why.** If you decide what counts as success *after* seeing results, you can always find a story that fits. Researchers call this "HARKing" (Hypothesizing After the Results are Known).
- Committing the rules first, with a git timestamp, proves they came first.
- Changes are allowed only as **dated amendments** at the bottom of the file, with a reason. The original text is never edited.

**Where.**
- The hypotheses and their support criteria are in the table in PREREGISTRATION.md.
- The constants are mirrored in `src/config.py`, so the code can't silently drift from the document.

**Viva questions**
- *Why pre-register for a course project?* It shows the conclusions weren't tuned to the results. The commit date (Oct 29, 19:06 IST) is before any model existed.
- *What if a hypothesis turns out false?* It gets reported as a finding. The proposal and rubric both say a negative result earns full credit if the evidence is sound.

---

## 2. Getting the data reproducibly (Oct 3)

**What.** `src/download_data.py` downloads BANKING77 `train.csv` and `test.csv` from PolyAI's GitHub at a **pinned commit** (`src/download_data.py:11`). It then checks each file's **SHA-256 checksum** (`src/download_data.py:19`).

**Why.**
- If the source repo changes later, the pinned commit still gives the same files.
- The checksum proves the files weren't corrupted or altered. Anyone who reruns the script gets byte-identical data.
- The Hugging Face loader for this dataset is deprecated, so GitHub is the reliable source.

**Data facts (verified):**
- 10,003 training rows and 3,080 test rows (40 per intent).
- 77 intents, CC BY 4.0 license.
- Two columns: `text` and `category`.

**Viva questions**
- *Why not commit the raw CSVs?* The script plus checksum makes them reproducible without duplicating the dataset. `data/raw/` is gitignored. The processed split *is* committed so graders can see the exact folds.

---

## 3. Repository hygiene (Oct 29)

**What.**
- `.gitignore` excludes the virtual environment, raw data, personal notes (`notes/`, `PRD.md`, `CLAUDE.md`), and any `.env` file.
- The early git history was rewritten so the personal planning files never appear on GitHub. The pre-registration commit kept its original date.

**Why.**
- The course requires that no credentials or private data be published.
- The project uses no API keys at all. `.env` is ignored anyway, as a safety net.

---

## 4. Packages (Oct 30)

**What.** `requirements.txt` pins exact versions: scikit-learn 1.9.1, numpy 2.5.4, pandas 3.1.0, scipy 1.18.1, matplotlib 3.11.2, plus jupyter and pytest. All were tested on Python 3.14.

**Why.** Results can change between library versions, for example when default parameters change. Pinning makes the results reproducible.

---

## 5. Text normalization and features (Oct 30)

**What.** `src/features.py` defines the following.

| Piece | Code | Choice | Reason |
|---|---|---|---|
| Normalization | `src/features.py:15` | Lowercase, collapse whitespace, strip | "Card" and "card" should be the same word. Extra spaces carry no meaning. |
| Tokenizer | `src/features.py:11` | Words of 2 or more letters or digits (sklearn's default pattern) | Simple, and easy to copy exactly into JavaScript for the website. |
| Word TF-IDF | `src/features.py:20` | 1–2 word n-grams, `sublinear_tf=True` | Bigrams capture phrases like "not arrived" and "top up". Sublinear TF uses 1 + log(count), so a word repeated 3 times doesn't count 3 times as much. |
| Character TF-IDF | `src/features.py:32` | 3–5 character n-grams within words | Robust to typos and word variants ("top-up" / "topup" / "top ups"). |
| Stopwords | kept | Not removed | Words like "not", "why" and "how" change the intent. For example, "card *not* working" and "card working" mean different things. Removing them is planned as an ablation. |

**TF-IDF in one line.** A word's weight is (how often it appears in this message) × log(1 / how many messages contain it). Common words get low weight, and distinctive words get high weight. Each message vector is then scaled to length 1.

**Why the tokenizer must be shared.** The website runs the model in the browser. If the JavaScript splits text differently from Python, the site would show different predictions from the ones reported. A parity test (planned) checks they match.

**Viva questions**
- *Why `lowercase=False` in the vectorizer?* Normalization already lowercases. Doing it once, in one function, keeps Python and JavaScript identical.
- *Is fitting TF-IDF leakage?* Only if it's fit on validation or test data. In modeling, it's always fit **inside each training fold**, using an sklearn `Pipeline`.

---

## 6. The split (Oct 30)

**File:** `src/split.py`. Run it with `python -m src.split` (about 20 s). Outputs go to `data/processed/`.

### Step-by-step

| Step | Code | What happens | Why |
|---|---|---|---|
| 1. Normalize | `src/split.py:72` | Add a `norm_text` column | So duplicates are detected regardless of case or spacing. |
| 2. Exact duplicates | `src/split.py:74` | Drop repeated `norm_text`, keeping the first copy. **4 removed, 9,999 rows left.** No duplicate had conflicting labels. | A repeated message counts twice and can land in both training and validation folds. |
| 3. Near-duplicate pairs | `src/split.py:37` | Compute TF-IDF vectors, then the cosine similarity of every pair. Keep pairs ≥ 0.90. The work is done in chunks of 2,000 rows to save memory (`src/split.py:44` keeps each pair once, i < j). | Paraphrases that differ by one word ("top up" vs "top-up") are effectively the same example. |
| 4. Groups | `src/split.py:51` | Treat pairs as edges in a graph. Each **connected component** becomes a group. If A≈B and B≈C, then A, B and C share a group. | Ensures a whole cluster of near-copies stays together. |
| 5. Folds | `src/split.py:57` | `StratifiedGroupKFold`, 5 folds, run for 5 seeds (335–339). | See below. |
| 6. Unseen intents | `src/split.py:65` | Pick 10 intents at random (seed 2026) and flag their rows `is_unseen`. | For the "can it reject intents it never saw?" experiment. |
| 7. Report | `src/split.py:102` onward | Counts and checks are written to `split_report.json`. | Evidence for the website's Data section. |

### Why `StratifiedGroupKFold`

- **Group:** messages in the same near-duplicate group always go to the **same fold**. If a near-copy were in the training folds while its twin was being validated, the model would just "recognize" it. That inflates the validation score, which is a form of leakage.
- **Stratified:** each fold keeps roughly the same intent proportions. The smallest intent has only 35 examples. Without stratification, a fold could end up with almost none of them. The tests confirm every intent has **at least 7 examples in every fold**.

### Why 5 seeds

The fold assignment is random. Running the whole evaluation with 5 different shuffles shows how much the scores move just from *which* messages land in which fold. `fold_335` is the main one. `fold_336`–`fold_339` are only for measuring this spread.

### Output columns (`data/processed/train_split.csv`)

| Column | Meaning |
|---|---|
| `text`, `category` | Original message and label |
| `norm_text` | Normalized text |
| `group` | Near-duplicate group at 0.90 |
| `fold_335` … `fold_339` | Fold number (0–4) for each seed |
| `group_sens`, `fold_sens` | Groups and folds at the stricter 0.80 threshold (section 7) |
| `is_unseen` | True for the 10 held-out intents |

### Results

| Measure | Value |
|---|---|
| Rows after removing duplicates | 9,999 |
| Near-duplicate pairs at 0.90 | 153 |
| Groups | 9,854. Most are single messages; 275 rows are in multi-message groups. |
| Largest group | 4 messages |
| Fold sizes | 1,999 / 2,000 / 2,000 / 2,000 / 2,000 |
| Minimum examples of any intent in any fold | 7 |
| Unseen intents | activate_my_card, automatic_top_up, card_arrival, declined_card_payment, declined_cash_withdrawal, exchange_via_app, passcode_forgotten, pending_top_up, top_up_by_card_charge, wrong_exchange_rate_for_cash_withdrawal (1,414 rows) |

**Viva questions**
- *The near-duplicate TF-IDF is fit on the whole training set. Isn't that leakage?* No. It's only used to decide **which rows must stay together**. It never becomes a model input, and it never uses labels or test data. The model's own TF-IDF is refit inside each training fold.
- *Why not just use the official train/test split and a random validation set?* The test set is reserved for one final run. Five-fold CV gives a prediction for *every* training message, which H1–H4 need (for example, a confusion matrix over all 9,999 messages), and it gives a sense of variance.
- *Why remove exact duplicates instead of grouping them?* An exact copy adds no new information and would get double weight in training. Near-duplicates are kept, since they're legitimate paraphrases, but grouped.

---

## 7. Amendment: near-duplicate sensitivity check (Oct 29)

**What happened.** Inspecting the pairs showed that the pre-registered 0.90 threshold misses obvious near-copies:

| Cosine | Message A | Message B |
|---|---|---|
| 0.899 | why do we need to verify top-up? | why do we need to verify top-ups? |
| 0.898 | how do i get a card if i live in the us? | can i get a card if i live in the us? |

At 0.80, pairs are still mostly one-word swaps ("charged multiple times for one *transaction*" vs "one *purchase*").

**The decision.**
- 0.90 was **not** changed to 0.80. Lowering a pre-registered threshold after looking at the data is exactly the kind of after-the-fact tweak pre-registration is meant to prevent.
- Instead, a **sensitivity check** was added. A second grouping at 0.80 (`group_sens`, `fold_sens`, `src/split.py:82` and `:94`) will be used to report the main model's CV score under both groupings.
- It's recorded as a dated amendment in PREREGISTRATION.md and was added **before any model was trained**. The constant is `SENSITIVITY_NEAR_DUP_COSINE` in `src/config.py`.

**How to interpret it later.**
- If the CV scores under 0.90 and 0.80 are close, near-duplicates aren't inflating the results.
- If 0.80 is clearly lower, near-copies were helping, and the 0.80 number is the more honest one. That would be reported openly.

**Results at 0.80:** 590 pairs, 9,474 groups, largest group 7. So stricter grouping doesn't chain unrelated messages together.

**Viva questions**
- *Isn't changing things after looking at the data cheating?* Changing a **rule** would be. Adding a **robustness check** without changing the primary rule, before any model results exist, is standard practice. The amendment is dated and explains why.

---

## 8. First findings from the split (Oct 30)

Some near-identical messages carry **different labels**: 4 pairs at 0.90 and 39 at 0.80. They're stored in `split_report.json` → `near_duplicate_pairs_with_different_labels`.

| Message A | Label A | Message B | Label B |
|---|---|---|---|
| has my top up gone through? | pending_top_up | has my top-up gone through? | top_up_reverted |
| i transferred some money but it hasn't arrived. | transfer_not_received_by_recipient | i transferred some money but it hasn't arrived yet | balance_not_updated_after_bank_transfer |
| what is this $1 charge on my account? | extra_charge_on_statement | what is this charge on my card? | direct_debit_payment_not_recognised |
| how can i get new card? | contactless_not_working | can i get a new card? | order_physical_card |

**Why this matters for H1** ("most errors come from overlapping intent definitions, not weak models"):
- No model can get both messages in such a pair right, because the text is effectively the same.
- So part of the error rate is set by the labels, not by the model.
- The last row looks like a possible **mislabel** ("how can i get new card?" → `contactless_not_working`). That's one of the error causes in the planned hand-labeled error taxonomy.

**Caution.** This is anecdotal evidence, not yet a test of H1. H1 is decided by the pre-registered criteria using OOF errors.

---

## 9. Leakage guard and tests (Oct 30)

**Leakage guard** (`tests/test_leakage_guard.py`)
- **What it does:** fails if any module other than `evaluate_final.py` (plus `config.py` and `download_data.py`, which only define or download the file) refers to the test set.
- **The bug that was fixed:** the first version searched for the text "test.csv" anywhere in a file. It flagged docstrings such as *"Never reads test.csv"*, which was a false alarm.
- **The fix** (`tests/test_leakage_guard.py:13`): parse the Python code into a syntax tree (`ast`) and look only for real code references. That means the `TEST_CSV` name or a string literal ending in `test.csv`, not comments or docstrings.
- A second test (`tests/test_leakage_guard.py:28`) proves the guard still catches a real reference. A guard that can never fail would be useless.

**Split tests** (`tests/test_split.py`)

| Test | Checks |
|---|---|
| `test_no_exact_duplicates_remain` | Deduplication worked |
| `test_groups_never_cross_folds` | No near-duplicate group is split across folds, for both 0.90 and 0.80 |
| `test_every_intent_in_every_fold` | All 77 intents appear in all 5 folds, for every seed |
| `test_sensitivity_groups_are_coarser` | Every 0.90 group sits inside a single 0.80 group |
| `test_unseen_intents` | Exactly 10 unseen intents, correctly flagged |
| `test_split_is_reproducible` | Running the split twice gives identical output |

**Status:** 9/9 tests pass (`python -m pytest`, about 35 s).

**Viva questions**
- *How do you know the test set wasn't used?* Only one script can load it, a test enforces that, and that script runs once, on Nov 12, after models are frozen.
