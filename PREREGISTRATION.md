# Pre-registration: hypotheses, split, and decision rules

Committed **before any model comparison and before the test set is used**. The git history shows the date.
After this commit, changes are only added as dated entries under **Amendments** at the bottom. Earlier text is never edited.

## Research question

Why do classical ML intent classifiers fail on banking customer queries, and when should they abstain instead of answering? The data is BANKING77 (77 intents), and only methods from the ES 335 course are used.

## Hypotheses and what counts as support

All decisions on H1–H4 are made from **out-of-fold (OOF) predictions** on the training data. The test set only confirms them once, at the end.

| ID | Hypothesis | Supported if |
|---|---|---|
| **H1** | Most errors come from overlapping intent definitions, not from weak models. | (a) ≥ 50% of the main model's OOF errors fall *within* confusable intent groups. These groups are built from the OOF confusion matrix (described below). (b) Among about 50 hand-labeled errors, the most common cause is "ambiguous label" or "vague message", not "vocabulary gap". (c) The top-10 confused pairs overlap substantially across model families (LR, KNN, random forest): at least 5 of 10 are shared between LR and each other family. |
| **H2** | Confidence-based abstention beats random abstention at every coverage level. | The selective error of max-probability abstention is below the error of random abstention at every coverage level in {0.50, 0.55, …, 0.95}. The 95% bootstrap CI of the difference must exclude 0 at every level. |
| **H3** | The most accurate model is not necessarily the best calibrated. | The model with the highest OOF macro-F1 does **not** have the lowest OOF ECE (15 equal-width bins, uncalibrated probabilities). Here, "different" means the bootstrap CIs of the ECE values don't overlap. |
| **H4** | Low-dimensional SVD features keep most of the signal in the full TF-IDF features. | LR on TruncatedSVD features with **≤ 300 dimensions** reaches ≥ 90% of the macro-F1 of LR on the full TF-IDF features (OOF). |
| **U** (exploratory) | Confidence can reject messages from intents the model never saw. | Reported, not hypothesis-tested: (1) the rejection rate of unseen-intent messages at the cost-chosen threshold, and (2) AUROC of max probability for separating seen from unseen intents. |

A hypothesis that turns out false is reported as a finding. The criteria above are not changed after the results are seen.

## Split and data rules

1. **The test set is locked.** `data/raw/test.csv` (3,080 rows) is read only by `src/evaluate_final.py`, and only in one run after all models are frozen.
2. **Exact duplicates** in the training data are removed. Before comparing, text is lowercased and whitespace collapsed.
3. **Near-duplicates:** two training messages whose TF-IDF cosine similarity is **≥ 0.90** are linked. TF-IDF here uses word 1–2 grams and is fit on the training data only. The connected components of these links form groups.
4. **Cross-validation:** `StratifiedGroupKFold`, 5 folds, stratified by intent, grouped by near-duplicate component. All tuning and model selection use it.
5. **Train/test overlap:** test messages with cosine ≥ 0.90 to any training message are counted. Test results are reported both with and without them. The test set itself is never altered.
6. All learned preprocessing (vocabulary, IDF, scaling, SVD, calibration) is fit inside each training fold through an sklearn `Pipeline`.

## Derived experiments

- **Unseen intents:** 10 intents are drawn at random with seed `UNSEEN_SEED` and held out of training. A 67-intent model is trained on the rest.
  - Messages from the held-out intents in the training file are the "unknown" validation data.
  - Messages from the held-out intents in the test file are the "unknown" test data.
- **Learning curves:** nested, stratified subsets of the training data (5%, 10%, 25%, 50%, 100%), each with 5 random draws. Every size is scored with the same CV folds. Test scores at each size are computed only in the single final run, and no decision is based on them.
- **Error predictor:** a logistic regression that predicts "the main model is wrong". Its features are max probability, the margin between the top two probabilities, entropy, and message length. It's trained **only on OOF predictions** and compared against the plain max-probability rule.
- **Coarse categories:** the author groups the 77 intents into a few broad categories (cards, transfers, top-ups, exchange, identity/security, other). The grouping is documented in `reports/coarse_groups.json` **before** coarse-level accuracy is computed. Coarse accuracy and top-3 accuracy support H1.

## Abstention and cost

- **Cost per message:** `k × (wrong auto-answers) + 1 × (handoffs)`, with k ∈ {2, 5, 10}.
- **Threshold choice:** for each k, the threshold that minimizes cost on OOF predictions is chosen, then applied to the test set once.
- The cost ratio is an **assumption** and is labeled as one everywhere.

## Fixed numbers

| Constant | Value |
|---|---|
| Main seed | 335 |
| Seeds for seed spread | 335, 336, 337, 338, 339 |
| Unseen-intent seed | 2026 |
| CV folds | 5 |
| Near-duplicate cosine threshold | 0.90 |
| Bootstrap resamples | 1,000 (95% percentile CIs) |
| ECE bins | 15, equal-width |
| Cost ratios k | 2, 5, 10 |
| Learning-curve fractions | 0.05, 0.10, 0.25, 0.50, 1.00 (5 draws each) |

## Amendments

_None yet._
