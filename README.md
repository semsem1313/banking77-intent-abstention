# Knowing When to Ask a Human: Banking Intent Classification with Calibrated Handoff

ES 335 · Machine Learning 2026 course project.

**Research question:** why do classical machine-learning intent classifiers fail on banking customer queries, and when should they abstain instead of answering? Four hypotheses are tested:

- **H1:** most errors come from overlapping intent definitions, not from weak models.
- **H2:** confidence-based abstention beats random abstention at every coverage level.
- **H3:** the most accurate model is not necessarily the best calibrated.
- **H4:** low-dimensional SVD features keep most of the signal in the full TF-IDF features.

An exploratory test also asks whether the model can reject intents it never saw in training.

The hypotheses, split rules, and decision criteria were pre-registered in [PREREGISTRATION.md](PREREGISTRATION.md) before any model was trained. A step-by-step record of what was done and why is in [PROJECT_JOURNAL.md](PROJECT_JOURNAL.md).

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
.venv/Scripts/python -m src.tune             # (in progress)
.venv/Scripts/python -m src.evaluate_final   # run once, on frozen models
.venv/Scripts/python -m src.export_web
.venv/Scripts/python -m pytest               # ~35 s
```

## Status

| Step | State |
|---|---|
| Pre-registration, data download, split, tests | Done |
| Baselines, main model, OOF predictions | Next |
| Hypothesis analyses, final test run, website | To do |

## AI-use disclosure

Claude (Anthropic) was used for brainstorming, planning, and coding assistance. All results were checked by the author.

## Citations

Casanueva, I., Temčinas, T., Gerz, D., Henderson, M., & Vulić, I. (2020). *Efficient Intent Detection with Dual Sentence Encoders.* Proceedings of the 2nd Workshop on NLP for ConvAI, ACL.
