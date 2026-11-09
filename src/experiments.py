"""Unseen-intent test (U) and learning curves, both with the main model's configuration. Never reads test.csv.

U: train on the 67 seen intents only (rows with is_unseen == False), using the same folds.
   Seen held-out messages give the usual OOF predictions. Messages from the 10 unseen intents
   are predicted by the fold model whose fold they're in (no model ever trains on them).
   Reported: AUROC of max probability for separating seen from unseen, and the share of
   unseen messages handed off at the cost-chosen threshold (chosen on seen OOF only).

Learning curves: nested, stratified subsets of each training fold (5%-100%), 5 random draws
   per size, scored on the same held-out folds. "Nested" means the 10% subset contains the 5%
   subset, so the curves show adding data rather than reshuffling it.

Usage: python -m src.experiments  (writes reports/unseen.json, reports/learning_curve.json, figures)
"""
import json
import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from src.analysis import cost_threshold
from src.config import COST_RATIOS, DATA_PROCESSED, LEARNING_CURVE_DRAWS, LEARNING_CURVE_FRACTIONS, REPORTS, SEED
from src.metrics import macro_f1
from src.models import build_m1

MAIN = {"penalty": "L2", "features": "word+char", "C": 10}


def build_main():
    return build_m1(MAIN["penalty"], MAIN["features"], MAIN["C"])


# ---------- U: unseen intents ----------

def _unseen_fold(texts, labels, seen, folds, k):
    train = seen & (folds != k)
    model = build_main().fit(texts[train], labels[train])
    held = folds == k
    return k, model.predict_proba(texts[held]), model.classes_


def unseen_experiment(df):
    texts, labels = df["norm_text"].to_numpy(), df["category"].to_numpy()
    seen, folds = ~df["is_unseen"].to_numpy(), df[f"fold_{SEED}"].to_numpy()
    parts = Parallel(n_jobs=5)(delayed(_unseen_fold)(texts, labels, seen, folds, k) for k in sorted(np.unique(folds)))
    classes = parts[0][2]
    assert all(list(p[2]) == list(classes) for p in parts) and len(classes) == 67
    proba = np.zeros((len(df), len(classes)))
    for k, p, _ in parts:
        proba[folds == k] = p
    conf, pred = proba.max(axis=1), classes[proba.argmax(axis=1)]

    seen_correct = pred[seen] == labels[seen]
    thresholds = {k: cost_threshold(conf[seen], seen_correct, k) for k in COST_RATIOS}
    unseen_conf = conf[~seen]
    by_intent = {}
    for intent in sorted(set(labels[~seen])):
        m = labels == intent
        top = pd.Series(pred[m]).value_counts().head(3)
        by_intent[intent] = {"n": int(m.sum()), "mean_confidence": float(conf[m].mean()),
                             "handed_off_at_k5": float(np.mean(conf[m] < thresholds[5]["threshold"])),
                             "most_common_predictions": {str(a): int(b) for a, b in top.items()}}
    return {
        "n_seen": int(seen.sum()), "n_unseen": int((~seen).sum()),
        "seen_oof_accuracy": float(seen_correct.mean()),
        "seen_oof_macro_f1": macro_f1(labels[seen], pred[seen]),
        "mean_confidence": {"seen": float(conf[seen].mean()), "unseen": float(unseen_conf.mean())},
        "auroc_seen_vs_unseen": float(roc_auc_score(seen.astype(int), conf)),
        "at_cost_thresholds": {str(k): {"threshold": t["threshold"],
                                        "unseen_handed_off": float(np.mean(unseen_conf < t["threshold"])),
                                        "seen_handed_off": float(1 - t["coverage"]),
                                        "seen_accuracy_on_answered": t["selective_accuracy"]}
                               for k, t in thresholds.items()},
        "by_unseen_intent": by_intent,
        "_conf": {"seen": conf[seen].tolist(), "unseen": unseen_conf.tolist()},
    }


# ---------- Learning curves ----------

def nested_order(labels_train, rng):
    """Per intent, a random order of its training rows. Taking the first ceil(f * n) rows of each
    intent gives stratified subsets that are nested as f grows."""
    order = {}
    for c in np.unique(labels_train):
        idx = np.flatnonzero(labels_train == c)
        order[c] = rng.permutation(idx)
    return order


def subset(order, fraction):
    return np.sort(np.concatenate([idx[: max(1, math.ceil(fraction * len(idx)))] for idx in order.values()]))


def _curve_task(texts, labels, folds, k, draw, fraction):
    train_idx = np.flatnonzero(folds != k)
    order = nested_order(labels[train_idx], np.random.default_rng(SEED + draw))
    chosen = train_idx[subset(order, fraction)]
    model = build_main().fit(texts[chosen], labels[chosen])
    held = folds == k
    return fraction, draw, k, model.predict(texts[held]), len(chosen)


def learning_curves(df):
    texts, labels, folds = df["norm_text"].to_numpy(), df["category"].to_numpy(), df[f"fold_{SEED}"].to_numpy()
    tasks = [(k, d, f) for f in LEARNING_CURVE_FRACTIONS
             for d in (range(LEARNING_CURVE_DRAWS) if f < 1.0 else [0]) for k in sorted(np.unique(folds))]
    out = Parallel(n_jobs=10)(delayed(_curve_task)(texts, labels, folds, k, d, f) for k, d, f in tasks)
    preds, sizes = {}, {}
    for f, d, k, p, n_train in out:
        preds.setdefault((f, d), np.empty(len(df), dtype=object))[folds == k] = p
        sizes.setdefault(f, []).append(n_train)
    curve = []
    for f in LEARNING_CURVE_FRACTIONS:
        scores = [macro_f1(labels, preds[(f, d)]) for d in sorted({d for ff, d in preds if ff == f})]
        curve.append({"fraction": f, "mean_training_messages": float(np.mean(sizes[f])),
                      "macro_f1_by_draw": scores, "mean": float(np.mean(scores)),
                      "sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0})
    return curve


def main():
    df = pd.read_csv(DATA_PROCESSED / "train_split.csv")
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)

    start = time.time()
    u = unseen_experiment(df)
    conf = u.pop("_conf")
    (REPORTS / "unseen.json").write_text(json.dumps(u, indent=2))
    print(f"U done in {time.time() - start:.0f}s: AUROC {u['auroc_seen_vs_unseen']:.3f}; seen acc {u['seen_oof_accuracy']:.3f}; "
          f"mean conf seen {u['mean_confidence']['seen']:.3f} vs unseen {u['mean_confidence']['unseen']:.3f}", flush=True)
    for k, v in u["at_cost_thresholds"].items():
        print(f"   k={k:>2}: threshold {v['threshold']:.3f} -> unseen handed off {v['unseen_handed_off']:.1%}, "
              f"seen handed off {v['seen_handed_off']:.1%}", flush=True)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bins = np.linspace(0, 1, 26)
    ax.hist(conf["seen"], bins=bins, density=True, alpha=0.6, label="seen intents (67)")
    ax.hist(conf["unseen"], bins=bins, density=True, alpha=0.6, label="unseen intents (10)")
    ax.axvline(u["at_cost_thresholds"]["5"]["threshold"], color="black", ls="--", lw=1, label="k=5 threshold")
    ax.set(xlabel="Max probability", ylabel="Density", title=f"U: confidence on seen vs unseen intents (AUROC {u['auroc_seen_vs_unseen']:.2f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "figures" / "unseen_confidence.png", dpi=150)
    plt.close(fig)

    start = time.time()
    curve = learning_curves(df)
    (REPORTS / "learning_curve.json").write_text(json.dumps(curve, indent=2))
    print(f"Learning curves done in {time.time() - start:.0f}s", flush=True)
    for c in curve:
        print(f"   {c['fraction']:.0%} (~{c['mean_training_messages']:.0f} msgs): macro-F1 {c['mean']:.4f} ± {c['sd']:.4f}", flush=True)
    plot_learning_curve(curve, REPORTS / "figures" / "learning_curve.png")


def plot_learning_curve(curve, path):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = [c["mean_training_messages"] for c in curve]
    ax.errorbar(x, [c["mean"] for c in curve], yerr=[c["sd"] for c in curve], fmt="o-", capsize=3)
    ax.set(xscale="log", xlabel="Training messages (log scale)", ylabel="OOF macro-F1",
           title="Learning curve: main model (5 draws per size)")
    ax.set_xticks(x, [f"{v:,.0f}\n({c['fraction']:.0%})" for v, c in zip(x, curve)])
    ax.minorticks_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
