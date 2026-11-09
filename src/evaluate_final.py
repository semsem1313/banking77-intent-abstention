"""THE ONLY script allowed to read test.csv. It runs ONCE, after every choice is frozen.

All settings come from the OOF reports written earlier (configs, thresholds, groups, cost
ratios), so nothing is chosen here. Every frozen model is refit on all 9,999 training messages,
and the official test set (3,080 messages) is scored once.

Safety:
  --dry-run  runs the whole script with training fold 0 standing in for the test set and the
             other folds for training. test.csv is never opened. Use it to catch bugs first.
  A real run writes reports/final/LOCK (commit hash + time) and refuses to run again.

Usage: python -m src.evaluate_final --dry-run   then   python -m src.evaluate_final
"""
import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

from src.analysis import cost_threshold, ece, ece_ci, h2_test, reliability_bins, selective_error, CURVE_COVERAGES
from src.baselines import KeywordRule, MajorityClass
from src.config import (COST_RATIOS, DATA_PROCESSED, LEARNING_CURVE_DRAWS, LEARNING_CURVE_FRACTIONS,
                        NEAR_DUP_COSINE, REPORTS, SEED, TEST_CSV)
from src.error_predictor import build_e1, confidence_features
from src.experiments import nested_order, subset
from src.features import normalize_text, word_tfidf
from src.h1 import merge_labels, share_within_groups
from src.metrics import macro_f1, paired_bootstrap_diff, score
from src.models import FEATURE_SETS, RF_PARAMS, SVD_DIMS, SVD_MAIN_DIM, SVD_MAX_DIM, build_knn, build_m1

FINAL = REPORTS / "final"
LOCK = FINAL / "LOCK"


def load_frozen():
    """Read every frozen choice from the OOF reports."""
    m1 = json.loads((REPORTS / "m1.json").read_text())["models"]
    more = json.loads((REPORTS / "m2_m4.json").read_text())
    h4 = json.loads((REPORTS / "h4.json").read_text())
    h2 = json.loads((REPORTS / "h2.json").read_text())["M1_L2"]
    unseen = json.loads((REPORTS / "unseen.json").read_text())
    h1 = json.loads((REPORTS / "h1.json").read_text())
    coarse = json.loads((REPORTS / "coarse_groups.json").read_text())
    assert coarse["approved"], "coarse groups must be approved before the final run"
    return {
        "M1_L2": m1["M1_L2"]["config"], "M1_L1": m1["M1_L1"]["config"],
        "knn": more["M2_KNN"]["config"],
        "svd_C": {c["d"]: c["C"] for c in h4["curve"]},
        "thresholds": {t["k"]: t["threshold"] for t in h2["cost_thresholds"]},
        "unseen_thresholds": {int(k): v["threshold"] for k, v in unseen["at_cost_thresholds"].items()},
        "unseen_intents": json.loads((DATA_PROCESSED / "unseen_intents.json").read_text()),
        "h1_groups": h1["a_within_group_share"]["groups_at_5"],
        "coarse": {c: g for g, members in coarse["groups"].items() for c in members},
    }


def get_data(dry_run):
    df = pd.read_csv(DATA_PROCESSED / "train_split.csv")
    if dry_run:
        fold = df[f"fold_{SEED}"]
        return df[fold != 0].reset_index(drop=True), df[fold == 0].reset_index(drop=True)
    test = pd.read_csv(TEST_CSV)
    test["norm_text"] = test["text"].map(normalize_text)
    return df, test


def overlap_flags(train_texts, test_texts):
    """Test messages with word 1-2 gram TF-IDF cosine >= NEAR_DUP_COSINE to any training message."""
    vec = word_tfidf(ngram_range=(1, 2)).fit(train_texts)
    A, B = vec.transform(test_texts), vec.transform(train_texts)
    best = np.zeros(A.shape[0])
    for s in range(0, A.shape[0], 1000):
        best[s:s + 1000] = (A[s:s + 1000] @ B.T).max(axis=1).toarray().ravel()
    return best >= NEAR_DUP_COSINE


def fit_svd_models(train, test, frozen, classes):
    feats = FEATURE_SETS["word+char"]()
    X_tr, X_te = feats.fit_transform(train["norm_text"]), feats.transform(test["norm_text"])
    svd = TruncatedSVD(n_components=SVD_MAX_DIM, random_state=SEED).fit(X_tr)
    Z_tr, Z_te = svd.transform(X_tr), svd.transform(X_te)
    y = train["category"].to_numpy()
    curve, probas = [], {}
    for d in SVD_DIMS:
        A_tr, A_te = normalize(Z_tr[:, :d]), normalize(Z_te[:, :d])
        lr = LogisticRegression(C=frozen["svd_C"][d], max_iter=3000).fit(A_tr, y)
        p = lr.predict_proba(A_te)
        curve.append({"d": d, "C": frozen["svd_C"][d], "proba": p})
        if d == SVD_MAIN_DIM:
            probas["M3_SVD_LR"] = p
            probas["M4_RF_SVD"] = RandomForestClassifier(**RF_PARAMS, random_state=SEED, n_jobs=-1).fit(A_tr, y).predict_proba(A_te)
            probas["M2_KNN_SVD"] = KNeighborsClassifier(n_neighbors=frozen["knn"]["k"], weights=frozen["knn"]["weights"],
                                                        metric="cosine", algorithm="brute").fit(A_tr, y).predict_proba(A_te)
    return probas, curve


def _curve_point(train, test, draw, fraction):
    order = nested_order(train["category"].to_numpy(), np.random.default_rng(SEED + draw))
    idx = subset(order, fraction)
    model = build_m1("L2", "word+char", 10).fit(train["norm_text"].to_numpy()[idx], train["category"].to_numpy()[idx])
    return fraction, draw, macro_f1(test["category"], model.predict(test["norm_text"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    out_dir = FINAL / ("dry_run" if args.dry_run else "")
    if not args.dry_run and LOCK.exists():
        sys.exit(f"Refusing to run: the final evaluation already ran ({LOCK.read_text().strip()}).")
    out_dir.mkdir(parents=True, exist_ok=True)

    frozen = load_frozen()
    train, test = get_data(args.dry_run)
    classes = np.array(sorted(train["category"].unique()))
    y_tr, y = train["category"].to_numpy(), test["category"].to_numpy()
    print(f"{'DRY RUN' if args.dry_run else 'FINAL RUN'}: train {len(train)}, test {len(test)}", flush=True)

    # ---- Fit every frozen model on all training data; predict the test set ----
    builders = {
        "M1_L2": lambda: build_m1("L2", frozen["M1_L2"]["features"], frozen["M1_L2"]["C"]),
        "M1_L1": lambda: build_m1("L1", frozen["M1_L1"]["features"], frozen["M1_L1"]["C"]),
        "M2_KNN": lambda: build_knn(frozen["knn"]["k"], frozen["knn"]["weights"]),
    }
    fitted = Parallel(n_jobs=3)(delayed(lambda b: b().fit(train["norm_text"], y_tr))(b) for b in builders.values())
    probas = {name: m.predict_proba(test["norm_text"]) for name, m in zip(builders, fitted)}
    for m in fitted:
        assert list(m.classes_) == list(classes)
    svd_probas, svd_curve = fit_svd_models(train, test, frozen, classes)
    probas.update(svd_probas)
    preds = {name: classes[p.argmax(axis=1)] for name, p in probas.items()}
    preds["B0_majority"] = MajorityClass().fit(train["norm_text"], y_tr).predict(test["norm_text"])
    for k in (10, 20):
        preds[f"B1_keyword_k{k}"] = KeywordRule(k).fit(train["norm_text"], y_tr).predict(test["norm_text"])
    print("models fitted", flush=True)

    result = {"mode": "dry_run" if args.dry_run else "final", "n_train": len(train), "n_test": len(test)}

    # ---- Headline scores, with and without train/test near-duplicate overlap ----
    overlap = overlap_flags(train["norm_text"], test["norm_text"])
    result["overlap"] = {"n_overlapping_test_messages": int(overlap.sum()), "share": float(overlap.mean())}
    result["scores"] = {name: score(y, p) for name, p in preds.items()}
    result["scores_without_overlap"] = {name: score(y[~overlap], preds[name][~overlap], with_ci=False)
                                        for name in ("M1_L2", "B1_keyword_k20")}
    result["M1_L2_vs_B1_keyword_k20_macro_f1"] = paired_bootstrap_diff(y, preds["M1_L2"], preds["B1_keyword_k20"], macro_f1)
    print("scores done", flush=True)

    # ---- H2: abstention, and the frozen OOF thresholds applied to the test set ----
    conf = probas["M1_L2"].max(axis=1)
    correct = preds["M1_L2"] == y
    applied = []
    for k, t in frozen["thresholds"].items():
        answered = conf >= t
        n_wrong = int((answered & ~correct).sum())
        applied.append({"k": k, "threshold": t, "coverage": float(answered.mean()),
                        "selective_accuracy": float(correct[answered].mean()) if answered.any() else None,
                        "cost_per_message": float((k * n_wrong + (~answered).sum()) / len(y)),
                        "cost_answer_all": float(k * (~correct).mean()),
                        "test_optimal_cost_for_reference": cost_threshold(conf, correct, k)["cost_per_message"]})
    result["H2"] = {"test": h2_test(conf, correct), "frozen_thresholds_on_test": applied,
                    "aurc": float(np.mean([selective_error(conf, correct, c) for c in CURVE_COVERAGES])),
                    "aurc_random": float(1 - correct.mean()),
                    "curve": [{"coverage": float(c), "selective_error": selective_error(conf, correct, c)} for c in CURVE_COVERAGES]}

    # ---- E1: fit on all OOF features from training, score on the test set ----
    oof = np.load(REPORTS / "oof" / "M1_L2.npz", allow_pickle=True)
    full_train = pd.read_csv(DATA_PROCESSED / "train_split.csv")
    if args.dry_run:
        keep = full_train[f"fold_{SEED}"] != 0
        oof_proba, oof_y, oof_text = oof["proba"][keep], full_train["category"][keep], full_train["norm_text"][keep]
    else:
        oof_proba, oof_y, oof_text = oof["proba"], full_train["category"], full_train["norm_text"]
    oof_wrong = oof["classes"][np.asarray(oof_proba).argmax(axis=1)] != oof_y.to_numpy()
    e1 = build_e1().fit(confidence_features(np.asarray(oof_proba, dtype=float), oof_text).to_numpy(), oof_wrong)
    e1_conf = 1 - e1.predict_proba(confidence_features(probas["M1_L2"], test["norm_text"]).to_numpy())[:, 1]
    result["E1"] = {"auroc": {"max_prob": float(roc_auc_score(~correct, -conf)), "e1": float(roc_auc_score(~correct, -e1_conf))},
                    "aurc": {"max_prob": result["H2"]["aurc"],
                             "e1": float(np.mean([selective_error(e1_conf, correct, c) for c in CURVE_COVERAGES]))}}

    # ---- H3: calibration of every probabilistic model on the test set ----
    result["H3"] = {}
    for name, p in probas.items():
        c, ok = p.max(axis=1), preds[name] == y
        result["H3"][name] = {"macro_f1": macro_f1(y, preds[name]), "ece": ece(c, ok), "ece_ci95": ece_ci(c, ok),
                              "mean_confidence": float(c.mean()), "accuracy": float(ok.mean()),
                              "reliability": reliability_bins(c, ok)}
    print("H2, E1, H3 done", flush=True)

    # ---- H4: macro-F1 vs SVD dimension on the test set ----
    full_f1 = macro_f1(y, preds["M1_L2"])
    result["H4"] = {"full_tfidf_macro_f1": full_f1,
                    "curve": [{"d": c["d"], "C": c["C"], "macro_f1": macro_f1(y, classes[c["proba"].argmax(axis=1)]),
                               "share_of_full_tfidf": macro_f1(y, classes[c["proba"].argmax(axis=1)]) / full_f1} for c in svd_curve]}

    # ---- H1 on the test set, using the groups fixed from OOF ----
    groups = frozen["h1_groups"]
    share, inside, total = share_within_groups(y, preds["M1_L2"], groups)
    order = np.argsort(-probas["M1_L2"], axis=1)[:, :3]
    yc = np.array([frozen["coarse"][c] for c in y])
    pc = np.array([frozen["coarse"][c] for c in preds["M1_L2"]])
    result["H1"] = {"share_of_errors_inside_oof_groups": share, "errors_inside": inside, "errors_total": total,
                    "merge": {"accuracy_before": float(correct.mean()),
                              "accuracy_after": float(np.mean(merge_labels(y, groups) == merge_labels(preds["M1_L2"], groups))),
                              "macro_f1_after": macro_f1(merge_labels(y, groups), merge_labels(preds["M1_L2"], groups))},
                    "top3_accuracy": float(np.mean([t in classes[o] for t, o in zip(y, order)])),
                    "coarse_accuracy": float(np.mean(yc == pc))}

    # ---- U: 67-intent model on the test set ----
    unseen_set = set(frozen["unseen_intents"])
    seen_tr = ~train["category"].isin(unseen_set).to_numpy()
    model67 = build_m1("L2", "word+char", 10).fit(train["norm_text"][seen_tr], y_tr[seen_tr])
    p67 = model67.predict_proba(test["norm_text"])
    conf67, is_seen = p67.max(axis=1), ~pd.Series(y).isin(unseen_set).to_numpy()
    result["U"] = {"auroc_seen_vs_unseen": float(roc_auc_score(is_seen.astype(int), conf67)),
                   "seen_accuracy": float(np.mean(model67.classes_[p67.argmax(axis=1)][is_seen] == y[is_seen])),
                   "mean_confidence": {"seen": float(conf67[is_seen].mean()), "unseen": float(conf67[~is_seen].mean())},
                   "at_frozen_thresholds": {str(k): {"threshold": t, "unseen_handed_off": float(np.mean(conf67[~is_seen] < t)),
                                                     "seen_handed_off": float(np.mean(conf67[is_seen] < t))}
                                            for k, t in frozen["unseen_thresholds"].items()}}
    print("H4, H1, U done", flush=True)

    # ---- Learning-curve points on the test set (no decision is based on these) ----
    tasks = [(d, f) for f in LEARNING_CURVE_FRACTIONS for d in (range(LEARNING_CURVE_DRAWS) if f < 1.0 else [0])]
    points = Parallel(n_jobs=10)(delayed(_curve_point)(train, test, d, f) for d, f in tasks)
    result["learning_curve"] = [{"fraction": f, "macro_f1_by_draw": [s for ff, _, s in points if ff == f],
                                 "mean": float(np.mean([s for ff, _, s in points if ff == f]))} for f in LEARNING_CURVE_FRACTIONS]

    # ---- Save, and lock a real run ----
    np.savez_compressed(out_dir / "test_M1_L2.npz", proba=probas["M1_L2"].astype(np.float32), classes=classes,
                        y=y, text=test["text"].to_numpy())
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    result["ran_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result["git_commit"] = commit
    (out_dir / "final_test.json").write_text(json.dumps(result, indent=2, default=float))
    if not args.dry_run:
        LOCK.write_text(f"ran at {result['ran_at_utc']} on commit {commit}\n")

    s = result["scores"]
    print(f"M1-L2 macro-F1 {s['M1_L2']['macro_f1']['value']:.4f} {[round(v, 4) for v in s['M1_L2']['macro_f1']['ci95']]}; "
          f"B1 k20 {s['B1_keyword_k20']['macro_f1']['value']:.4f}; overlap {result['overlap']['n_overlapping_test_messages']}")
    print(f"H2 supported={result['H2']['test']['supported']}; U AUROC {result['U']['auroc_seen_vs_unseen']:.3f}; "
          f"H1 share {share:.3f}; coarse {result['H1']['coarse_accuracy']:.3f}")


if __name__ == "__main__":
    main()
