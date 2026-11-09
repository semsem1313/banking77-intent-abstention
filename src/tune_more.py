"""Tune M2 (cosine KNN), M3 (SVD + logistic regression, the H4 curve), and M4 (random forest on SVD).

Never reads test.csv. Uses the main folds (fold_<SEED>) and saves OOF probabilities, like tune.py.

SVD is fit inside each training fold. One TruncatedSVD with SVD_MAX_DIM components is fit
per fold, and each smaller dimension d uses its first d components. Components come out
ordered by explained variance, so the first d approximate a d-component SVD, without refitting
it 7 times. Rows are L2-normalized after truncation (standard for LSA), so cosine and dot
product agree.

Usage: python -m src.tune_more  (writes reports/m2_m4.json, reports/h4.json, reports/oof/M2_*.npz, M3_*.npz, M4_*.npz)
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

from src.config import DATA_PROCESSED, REPORTS, SEED
from src.metrics import macro_f1, paired_bootstrap_diff, score
from src.models import (FEATURE_SETS, M2_GRID, RF_PARAMS, SVD_DIMS, SVD_LR_C, SVD_MAIN_DIM, SVD_MAX_DIM,
                        build_knn)
from src.tune import fold_scores, oof_proba, one_se_choice


def _svd_fold(texts, labels, train_mask, classes, knn_cfg):
    """Everything learned here (TF-IDF, SVD, classifiers) sees only the training folds."""
    feats = FEATURE_SETS["word+char"]()
    X_tr = feats.fit_transform(texts[train_mask])
    X_va = feats.transform(texts[~train_mask])
    svd = TruncatedSVD(n_components=SVD_MAX_DIM, random_state=SEED).fit(X_tr)
    Z_tr, Z_va = svd.transform(X_tr), svd.transform(X_va)
    out = {"explained_variance": svd.explained_variance_ratio_.tolist()}
    y_tr = labels[train_mask]
    for d in SVD_DIMS:
        A_tr, A_va = normalize(Z_tr[:, :d]), normalize(Z_va[:, :d])
        for C in SVD_LR_C:
            lr = LogisticRegression(C=C, max_iter=3000).fit(A_tr, y_tr)
            assert list(lr.classes_) == list(classes)
            out[("lr", d, C)] = lr.predict_proba(A_va)
        if d == SVD_MAIN_DIM:
            rf = RandomForestClassifier(**RF_PARAMS, random_state=SEED, n_jobs=3).fit(A_tr, y_tr)
            out[("rf", d)] = rf.predict_proba(A_va)
            knn = KNeighborsClassifier(n_neighbors=knn_cfg["k"], weights=knn_cfg["weights"], metric="cosine",
                                       algorithm="brute").fit(A_tr, y_tr)
            out[("knn", d)] = knn.predict_proba(A_va)
    return out


def assemble(parts, folds, key, n, n_classes):
    proba = np.zeros((n, n_classes))
    for k, part in parts.items():
        proba[folds == k] = part[key]
    return proba


def save_oof(name, proba, classes, folds):
    np.savez_compressed(REPORTS / "oof" / f"{name}.npz", proba=proba.astype(np.float32), classes=classes, fold=folds)


def main():
    df = pd.read_csv(DATA_PROCESSED / "train_split.csv")
    classes = np.array(sorted(df["category"].unique()))
    y = df["category"].to_numpy()
    folds = df[f"fold_{SEED}"].to_numpy()
    n = len(df)
    m1 = np.load(REPORTS / "oof" / "M1_L2.npz", allow_pickle=True)
    m1_pred = m1["classes"][m1["proba"].argmax(axis=1)]
    m1_macro_f1 = macro_f1(y, m1_pred)
    summary = {}

    # ---- M2: KNN on full TF-IDF ----
    rows, probas = [], {}
    for cfg in M2_GRID:
        start = time.time()
        proba = oof_proba(lambda cfg=cfg: build_knn(cfg["k"], cfg["weights"]), df, f"fold_{SEED}", classes)
        name = f"M2_KNN[k={cfg['k']},{cfg['weights']}]"
        rows.append({"name": name, **cfg, "macro_f1": fold_scores(y, classes[proba.argmax(axis=1)], folds),
                     "seconds": round(time.time() - start, 1)})
        probas[name] = proba
        print(f"{name:30s} macro-F1 {rows[-1]['macro_f1']['mean']:.4f} ± {rows[-1]['macro_f1']['sd']:.4f}", flush=True)
    chosen, detail = one_se_choice(rows)
    knn_cfg = {"k": chosen["k"], "weights": chosen["weights"]}
    save_oof("M2_KNN", probas[chosen["name"]], classes, folds)
    summary["M2_KNN"] = {"chosen": chosen["name"], "config": knn_cfg, "selection": detail, "grid": rows}
    print(f"M2: chose {chosen['name']} (best {detail['best']})", flush=True)

    # ---- SVD per fold: M3 (H4 curve), M4, and KNN on SVD ----
    texts = df["norm_text"].to_numpy()
    fold_ids = sorted(np.unique(folds))
    start = time.time()
    results = Parallel(n_jobs=len(fold_ids))(
        delayed(_svd_fold)(texts, y, folds != k, classes, knn_cfg) for k in fold_ids)
    parts = dict(zip(fold_ids, results))
    print(f"SVD folds done in {time.time() - start:.0f}s", flush=True)

    curve = []
    for d in SVD_DIMS:
        c_rows = []
        for C in SVD_LR_C:
            proba = assemble(parts, folds, ("lr", d, C), n, len(classes))
            pred = classes[proba.argmax(axis=1)]
            c_rows.append({"name": f"M3_SVD_LR[d={d},C={C}]", "C": C, "macro_f1": fold_scores(y, pred, folds),
                           "pooled_macro_f1": macro_f1(y, pred), "_proba": proba})
        best_c, _ = one_se_choice(c_rows)
        curve.append({"d": d, "C": best_c["C"], "macro_f1_fold_mean": best_c["macro_f1"]["mean"],
                      "macro_f1_fold_sd": best_c["macro_f1"]["sd"], "pooled_macro_f1": best_c["pooled_macro_f1"],
                      "share_of_full_tfidf": best_c["pooled_macro_f1"] / m1_macro_f1,
                      "explained_variance": float(np.mean([sum(p["explained_variance"][:d]) for p in results]))})
        if d == SVD_MAIN_DIM:
            save_oof("M3_SVD_LR", best_c["_proba"], classes, folds)
            summary["M3_SVD_LR"] = {"chosen": best_c["name"], "config": {"d": d, "C": best_c["C"]}}
        print(f"d={d:4d}: C={best_c['C']:>3}  macro-F1 {best_c['pooled_macro_f1']:.4f}  "
              f"({curve[-1]['share_of_full_tfidf']:.1%} of full TF-IDF)  var {curve[-1]['explained_variance']:.2f}", flush=True)

    save_oof("M4_RF_SVD", assemble(parts, folds, ("rf", SVD_MAIN_DIM), n, len(classes)), classes, folds)
    save_oof("M2_KNN_SVD", assemble(parts, folds, ("knn", SVD_MAIN_DIM), n, len(classes)), classes, folds)
    summary["M4_RF_SVD"] = {"config": {"d": SVD_MAIN_DIM, **RF_PARAMS}}
    summary["M2_KNN_SVD"] = {"config": {"d": SVD_MAIN_DIM, **knn_cfg}, "note": "Same k as M2; tests KNN in a lower-dimensional space."}

    # ---- Scores with CIs, and paired comparison against the main model ----
    for name in ("M2_KNN", "M2_KNN_SVD", "M3_SVD_LR", "M4_RF_SVD"):
        d = np.load(REPORTS / "oof" / f"{name}.npz", allow_pickle=True)
        pred = d["classes"][d["proba"].argmax(axis=1)]
        summary[name]["scores"] = score(y, pred, folds)
        summary[name]["vs_M1_L2_macro_f1"] = paired_bootstrap_diff(y, pred, m1_pred, macro_f1)
        print(f"{name:12s} macro-F1 {summary[name]['scores']['macro_f1']['value']:.4f} "
              f"{[round(v, 4) for v in summary[name]['scores']['macro_f1']['ci95']]}  "
              f"vs M1-L2 {summary[name]['vs_M1_L2_macro_f1']['diff']:+.4f}", flush=True)

    # ---- H4 verdict (pre-registered: some d <= 300 reaches >= 90% of full TF-IDF macro-F1) ----
    passing = [c["d"] for c in curve if c["d"] <= 300 and c["share_of_full_tfidf"] >= 0.90]
    h4 = {"full_tfidf_macro_f1": m1_macro_f1, "curve": curve, "dims_at_or_below_300_reaching_90pct": passing,
          "smallest_d_reaching_90pct": min((c["d"] for c in curve if c["share_of_full_tfidf"] >= 0.90), default=None),
          "supported": bool(passing)}
    (REPORTS / "h4.json").write_text(json.dumps(h4, indent=2))
    (REPORTS / "m2_m4.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(6, 4))
    ds = [c["d"] for c in curve]
    ax.plot(ds, [c["pooled_macro_f1"] for c in curve], "o-", label="SVD + logistic regression")
    ax.axhline(m1_macro_f1, color="black", lw=1, label=f"full TF-IDF + LR ({m1_macro_f1:.3f})")
    ax.axhline(0.9 * m1_macro_f1, color="grey", ls="--", lw=1, label="90% of full TF-IDF (H4 bar)")
    ax.axvline(300, color="grey", ls=":", lw=1)
    ax.set(xscale="log", xlabel="SVD dimensions d (log scale)", ylabel="OOF macro-F1", title="H4: how much signal survives compression")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "figures" / "h4_svd_dims.png", dpi=150)
    plt.close(fig)
    print(f"H4 supported={h4['supported']}  passing dims {passing}", flush=True)


if __name__ == "__main__":
    main()
