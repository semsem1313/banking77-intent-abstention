"""Shared metrics and bootstrap confidence intervals. Every model is scored with these."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src.config import N_BOOTSTRAP, SEED


def accuracy(y_true, y_pred):
    return float(accuracy_score(y_true, y_pred))


def macro_f1(y_true, y_pred):
    # zero_division=0: an intent that is never predicted gets F1 = 0 instead of a warning.
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


METRICS = {"accuracy": accuracy, "macro_f1": macro_f1}


def bootstrap_ci(y_true, y_pred, metric, n_resamples=N_BOOTSTRAP, seed=SEED, alpha=0.05):
    """Percentile bootstrap over messages: resample rows with replacement and recompute the metric."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = [metric(y_true[idx], y_pred[idx]) for idx in (rng.integers(0, n, n) for _ in range(n_resamples))]
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def score(y_true, y_pred, folds=None, with_ci=True):
    """Point estimate, 95% bootstrap CI, and (if folds given) the mean and SD across folds."""
    out = {}
    for name, metric in METRICS.items():
        entry = {"value": metric(y_true, y_pred)}
        if with_ci:
            entry["ci95"] = bootstrap_ci(y_true, y_pred, metric)
        if folds is not None:
            y_true_a, y_pred_a, folds_a = map(np.asarray, (y_true, y_pred, folds))
            per_fold = [metric(y_true_a[folds_a == k], y_pred_a[folds_a == k]) for k in np.unique(folds_a)]
            entry["per_fold"] = per_fold
            entry["fold_mean"] = float(np.mean(per_fold))
            entry["fold_sd"] = float(np.std(per_fold, ddof=1))
        out[name] = entry
    return out
