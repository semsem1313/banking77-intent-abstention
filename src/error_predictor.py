"""E1: predict when the main model (M1-L2) will be wrong. Never reads test.csv.

Features per message, all from M1-L2's out-of-fold probabilities (plus the text length):
  max_prob  - the top probability
  margin    - top probability minus the second
  entropy   - spread of the whole probability vector (high = unsure)
  length    - number of characters in the message
Target: 1 if M1-L2's OOF prediction is wrong.

E1 is itself scored out-of-fold on the same folds: it is fit on four folds' OOF features and
predicts the fifth. It never sees in-sample predictions, whose confidence would be inflated.

Question: does a learned error score abstain better than the plain max-probability rule?
Compared by AUROC for detecting errors and by AURC (area under the risk-coverage curve), each
with a paired bootstrap.

Usage: python -m src.error_predictor  (writes reports/e1.json)
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis import CURVE_COVERAGES, selective_error
from src.config import DATA_PROCESSED, N_BOOTSTRAP, REPORTS, SEED

FEATURES = ["max_prob", "margin", "entropy", "length"]


def confidence_features(proba, texts):
    top2 = np.sort(proba, axis=1)[:, -2:]
    p = np.clip(proba, 1e-12, 1)
    return pd.DataFrame({
        "max_prob": top2[:, 1],
        "margin": top2[:, 1] - top2[:, 0],
        "entropy": -(p * np.log(p)).sum(axis=1),
        "length": [len(t) for t in texts],
    })


def build_e1():
    # Features are on very different scales (length vs probabilities), so standardize first.
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


def oof_error_scores(X, wrong, folds):
    scores = np.zeros(len(X))
    for k in sorted(np.unique(folds)):
        train, held = folds != k, folds == k
        scores[held] = build_e1().fit(X[train], wrong[train]).predict_proba(X[held])[:, 1]
    return scores


def aurc(confidence, correct):
    return float(np.mean([selective_error(confidence, correct, c) for c in CURVE_COVERAGES]))


def paired_compare(conf_a, conf_b, wrong, n_resamples=N_BOOTSTRAP, seed=SEED):
    """Paired bootstrap of (A - B) for AUROC (higher is better) and AURC (lower is better)."""
    rng = np.random.default_rng(seed)
    n, correct = len(wrong), ~wrong
    d_auroc, d_aurc = [], []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        w, c = wrong[idx], correct[idx]
        d_auroc.append(roc_auc_score(w, -conf_a[idx]) - roc_auc_score(w, -conf_b[idx]))
        d_aurc.append(aurc(conf_a[idx], c) - aurc(conf_b[idx], c))
    q = lambda v: [float(x) for x in np.quantile(v, [0.025, 0.975])]
    return {"auroc_diff_ci95": q(d_auroc), "aurc_diff_ci95": q(d_aurc)}


def main():
    df = pd.read_csv(DATA_PROCESSED / "train_split.csv")
    d = np.load(REPORTS / "oof" / "M1_L2.npz", allow_pickle=True)
    proba, classes = d["proba"].astype(float), d["classes"]
    folds = df[f"fold_{SEED}"].to_numpy()
    wrong = classes[proba.argmax(axis=1)] != df["category"].to_numpy()

    X = confidence_features(proba, df["norm_text"]).to_numpy()
    e1_wrong = oof_error_scores(X, wrong, folds)
    e1_conf = 1 - e1_wrong            # E1's "confidence that M1 is right"
    maxprob_conf = X[:, 0]

    full = build_e1().fit(X, wrong)   # fit on everything only to *show* the coefficients
    coefs = dict(zip(FEATURES, full[-1].coef_[0].round(3).tolist()))

    result = {
        "n_errors": int(wrong.sum()),
        "auroc_error_detection": {"max_prob": float(roc_auc_score(wrong, -maxprob_conf)),
                                  "e1": float(roc_auc_score(wrong, -e1_conf))},
        "aurc": {"max_prob": aurc(maxprob_conf, ~wrong), "e1": aurc(e1_conf, ~wrong), "random": float(wrong.mean())},
        "e1_minus_max_prob": paired_compare(e1_conf, maxprob_conf, wrong),
        "standardized_coefficients_fit_on_all_for_display": coefs,
        "feature_means": {"correct": dict(zip(FEATURES, X[~wrong].mean(axis=0).round(3).tolist())),
                          "wrong": dict(zip(FEATURES, X[wrong].mean(axis=0).round(3).tolist()))},
    }
    a, b = result["auroc_error_detection"], result["aurc"]
    result["e1_minus_max_prob"].update({"auroc_diff": a["e1"] - a["max_prob"], "aurc_diff": b["e1"] - b["max_prob"]})
    (REPORTS / "e1.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
