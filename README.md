# Knowing When to Ask a Human: Banking Intent Classification with Calibrated Handoff

ES 335 · Machine Learning 2026 course project.

**Research question:** why do classical machine-learning intent classifiers fail on banking customer queries, and when should they abstain instead of answering? Four hypotheses are tested:

- **H1:** most errors come from overlapping intent definitions, not from weak models.
- **H2:** confidence-based abstention beats random abstention at every coverage level.
- **H3:** the most accurate model is not necessarily the best calibrated.
- **H4:** low-dimensional SVD features keep most of the signal in the full TF-IDF features.

An exploratory test also asks whether the model can reject intents it never saw in training.

The hypotheses, split rules, and decision criteria were pre-registered in [PREREGISTRATION.md](PREREGISTRATION.md) before any model was trained.

- **Live site:** https://semsem1313.github.io/banking77-intent-abstention/ (in progress)
- **Data:** BANKING77 (Casanueva et al., 2020, arXiv:2003.04807), CC BY 4.0. It's downloaded from [PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets) at a pinned commit, with SHA-256 checks. There are 10,003 training and 3,080 test messages across 77 intents.

## Evaluation design

- **The test set is used once.** `data/raw/test.csv` is read only by `src/evaluate_final.py`, in a single run after all models are frozen. A test (`tests/test_leakage_guard.py`) enforces this.
- **Training split** (`src/split.py`, outputs in `data/processed/`):
  - 4 exact duplicates are removed, leaving 9,999 messages.
  - Near-duplicates (word 1–2 gram TF-IDF cosine ≥ 0.90) are grouped into connected components (153 pairs).
  - Folds are 5-fold `StratifiedGroupKFold`, stratified by intent and grouped by near-duplicate component, for seeds 335–339. `fold_335` is the main assignment.
  - **Sensitivity check** (pre-registration amendment, 2026-10-29): a stricter grouping at cosine ≥ 0.80 (`fold_sens`). It's used to check that near-duplicates don't inflate CV scores.
  - 10 intents are held out at random (seed 2026) for the unseen-intent experiment.
- **All tuning, thresholds, and hypothesis decisions** use out-of-fold predictions from these folds.
- **All learned preprocessing** (TF-IDF, SVD, scaling, calibration) is fit inside each training fold.

## Reproduce

Tested on Python 3.14 (Windows).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m src.download_data
.venv/Scripts/python -m src.split            # ~20 s
.venv/Scripts/python -m src.baselines        # ~2.5 min (bootstrap CIs)
.venv/Scripts/python -m src.tune             # ~10 min on 16 cores (M1 grid + robustness checks)
.venv/Scripts/python -m src.evaluate_final   # run once, on frozen models
.venv/Scripts/python -m src.export_web
.venv/Scripts/python -m pytest               # ~35 s
```

## Status

| Step | State |
|---|---|
| Pre-registration, data download, split, tests | Done |
| Baselines (B0 majority, B1 keyword rule, B2 random abstention) | Done |
| Main model M1 (TF-IDF + logistic regression, L2 and L1), OOF probabilities | Done |
| H2 abstention, H3 calibration, error predictor, KNN, SVD (H4) | Next |
| H1 error analysis, unseen intents, learning curves, final test run, website | To do |

## Cross-validated results so far (out-of-fold, 9,999 training messages)

Each message is predicted by a model fit on the other four folds. Brackets show 95% bootstrap CIs. These numbers guide decisions; the final result is the single test-set run.

| Model | Accuracy | Macro-F1 |
|---|---|---|
| B0: always predict the majority intent | 0.019 [0.016, 0.021] | 0.001 [0.000, 0.001] |
| B1: keyword rule, 10 keywords per intent (primary) | 0.568 [0.558, 0.577] | 0.550 [0.540, 0.558] |
| B1: keyword rule, 20 keywords per intent (strongest baseline) | 0.649 [0.638, 0.658] | 0.629 [0.618, 0.637] |
| M1-L1: word+char TF-IDF, one-vs-rest L1 logistic regression, C=10 | 0.864 [0.857, 0.870] | 0.860 [0.852, 0.866] |
| **M1-L2: word+char TF-IDF, multinomial L2 logistic regression, C=10 (main model)** | **0.901 [0.896, 0.907]** | **0.900 [0.894, 0.906]** |

- **Versus the strongest baseline:** M1-L2 gains **+0.271** macro-F1 (paired bootstrap 95% CI [+0.263, +0.281]).
- **Selection:** C and the feature set were chosen by the one-standard-error rule over a 14-config grid (`reports/tuning/m1_grid.json`).
- **Robustness:**
  - Regrouping near-duplicates at 0.80 changes macro-F1 by −0.0008.
  - Across fold seeds 335–339, macro-F1 has mean 0.901 and SD 0.002.

B2 (random abstention) is applied to M1's predictions in the H2 analysis.

## AI-use disclosure

Claude (Anthropic) was used for brainstorming, planning, and coding assistance. All results were checked by the author.

## Citations

Casanueva, I., Temčinas, T., Gerz, D., Henderson, M., & Vulić, I. (2020). *Efficient Intent Detection with Dual Sentence Encoders.* Proceedings of the 2nd Workshop on NLP for ConvAI, ACL.
