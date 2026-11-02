"""Tune M1 with 5-fold StratifiedGroupKFold and save out-of-fold (OOF) probabilities. Never reads test.csv.

Protocol:
1. Every config in models.M1_GRID is scored by OOF macro-F1 on the main folds (fold_<SEED>).
2. Selection uses the one-standard-error rule: take the simplest config whose mean fold
   macro-F1 is within one standard error of the best. Grid order defines "simplest".
3. For each selected config, save the OOF probabilities and score them with bootstrap CIs.
   The main model (selected M1_L2) is also compared with the strongest baseline
   (paired bootstrap), rerun on the 0.80 sensitivity folds, and rerun on seeds 336-339.

The OOF score of the *selected* config is slightly optimistic, because it was also used
for the selection. The unbiased estimate comes from the single test-set run in evaluate_final.py.

Usage: python -m src.tune  (about 10 min; writes reports/tuning/m1_grid.json, reports/m1.json, reports/oof/*.npz)
"""
import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import log_loss

from src.config import DATA_PROCESSED, REPORTS, SEED, SEEDS
from src.metrics import macro_f1, paired_bootstrap_diff, score
from src.models import M1_GRID, build_m1, config_name

STRONGEST_BASELINE = "B1_keyword_k20"


def _fit_fold(build, texts, labels, train_mask, classes):
    model = build().fit(texts[train_mask], labels[train_mask])
    assert list(model.classes_) == list(classes), "every training fold must contain all intents"
    return model.predict_proba(texts[~train_mask])


def oof_proba(build, df, fold_col, classes):
    """OOF probability matrix (messages x intents). The 5 folds are fit in parallel."""
    texts, labels, folds = df["norm_text"].to_numpy(), df["category"].to_numpy(), df[fold_col].to_numpy()
    fold_ids = sorted(np.unique(folds))
    parts = Parallel(n_jobs=len(fold_ids))(
        delayed(_fit_fold)(build, texts, labels, folds != k, classes) for k in fold_ids
    )
    proba = np.zeros((len(df), len(classes)))
    for k, p in zip(fold_ids, parts):
        proba[folds == k] = p
    return proba


def fold_scores(y, pred, folds):
    per_fold = [macro_f1(y[folds == k], pred[folds == k]) for k in sorted(np.unique(folds))]
    return {"per_fold": per_fold, "mean": float(np.mean(per_fold)), "sd": float(np.std(per_fold, ddof=1))}


def one_se_choice(rows):
    """Simplest row (earliest in grid order) whose mean is within one SE of the best mean."""
    best = max(rows, key=lambda r: r["macro_f1"]["mean"])
    se = best["macro_f1"]["sd"] / np.sqrt(len(best["macro_f1"]["per_fold"]))
    threshold = best["macro_f1"]["mean"] - se
    chosen = next(r for r in rows if r["macro_f1"]["mean"] >= threshold)
    return chosen, {"best": best["name"], "best_mean": best["macro_f1"]["mean"], "se": float(se), "threshold": float(threshold)}


def main():
    df = pd.read_csv(DATA_PROCESSED / "train_split.csv")
    classes = np.array(sorted(df["category"].unique()))
    y = df["category"].to_numpy()
    fold_col = f"fold_{SEED}"
    folds = df[fold_col].to_numpy()
    (REPORTS / "tuning").mkdir(parents=True, exist_ok=True)
    (REPORTS / "oof").mkdir(parents=True, exist_ok=True)

    # 1. Grid search on OOF macro-F1.
    grid_rows, probas = {}, {}
    for family, configs in M1_GRID.items():
        penalty = family.split("_")[1]
        grid_rows[family] = []
        for cfg in configs:
            name = config_name(family, cfg)
            start = time.time()
            proba = oof_proba(lambda cfg=cfg: build_m1(penalty, cfg["features"], cfg["C"]), df, fold_col, classes)
            pred = classes[proba.argmax(axis=1)]
            row = {"name": name, **cfg, "macro_f1": fold_scores(y, pred, folds),
                   "log_loss": float(log_loss(y, proba, labels=classes)), "seconds": round(time.time() - start, 1)}
            grid_rows[family].append(row)
            probas[name] = proba
            print(f"{name:34s} macro-F1 {row['macro_f1']['mean']:.4f} ± {row['macro_f1']['sd']:.4f}  log-loss {row['log_loss']:.3f}  ({row['seconds']}s)", flush=True)
            (REPORTS / "tuning" / "m1_grid.json").write_text(json.dumps(grid_rows, indent=2))

    # 2. One-SE selection per family; save OOF probabilities; full scores.
    baselines = pd.read_csv(REPORTS / "oof" / "baselines.csv")
    assert (baselines["category"].to_numpy() == y).all(), "baseline OOF rows must align with the split"
    summary = {"selection_rule": "one standard error, simplest first", "models": {}}
    for family, rows in grid_rows.items():
        chosen, detail = one_se_choice(rows)
        proba = probas[chosen["name"]]
        pred = classes[proba.argmax(axis=1)]
        np.savez_compressed(REPORTS / "oof" / f"{family}.npz", proba=proba.astype(np.float32), classes=classes, fold=folds)
        summary["models"][family] = {
            "chosen": chosen["name"], "config": {"features": chosen["features"], "C": chosen["C"]}, "selection": detail,
            "scores": score(y, pred, folds), "log_loss": chosen["log_loss"],
            f"vs_{STRONGEST_BASELINE}_macro_f1": paired_bootstrap_diff(y, pred, baselines[STRONGEST_BASELINE].to_numpy(), macro_f1),
        }
        print(f"{family}: chose {chosen['name']} (best was {detail['best']})", flush=True)

    # 3. Robustness checks for the main model.
    main_cfg = summary["models"]["M1_L2"]["config"]
    build_main = lambda: build_m1("L2", main_cfg["features"], main_cfg["C"])
    main_value = macro_f1(y, classes[probas[summary["models"]["M1_L2"]["chosen"]].argmax(axis=1)])

    sens_pred = classes[oof_proba(build_main, df, "fold_sens", classes).argmax(axis=1)]
    sens_value = macro_f1(y, sens_pred)
    summary["sensitivity_near_dup_0.80"] = {"macro_f1_fold_0.90": main_value, "macro_f1_fold_0.80": sens_value,
                                            "difference": sens_value - main_value}
    seed_values = {SEED: main_value}
    for seed in SEEDS[1:]:
        seed_values[seed] = macro_f1(y, classes[oof_proba(build_main, df, f"fold_{seed}", classes).argmax(axis=1)])
    summary["seed_spread"] = {"macro_f1_by_seed": {str(k): v for k, v in seed_values.items()},
                              "mean": float(np.mean(list(seed_values.values()))),
                              "sd": float(np.std(list(seed_values.values()), ddof=1))}

    (REPORTS / "m1.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("sensitivity_near_dup_0.80", "seed_spread")}, indent=2))


if __name__ == "__main__":
    main()
