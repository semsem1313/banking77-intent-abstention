"""Export the main model and the results the website needs, into docs/assets/.

Never reads data/raw/test.csv. Test-set numbers come from reports/final/ (written once by
src/evaluate_final.py); everything else comes from training data or reports/*.json.

Outputs
  docs/assets/model.json   vocabularies, IDF weights, intercepts, class names, per-feature weight scales
  docs/assets/weights.bin  M1-L2 weights as int8, feature-major (feature j: 77 bytes, one per class)
  docs/assets/results.json every number and curve the site draws
  docs/assets/parity.json  500 training messages with Python's probabilities, for the browser parity check

Why int8: the full weight matrix is 77 x ~37k floats (11 MB as float32). Each feature's 77 weights
are scaled by their own max |w| / 127 and rounded. The rounding error per weight is at most
scale / 2, which changes the top prediction on very few messages; parity.json measures that.

Usage: python -m src.export_web
"""
import json

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, DOCS_ASSETS, REPORTS, SEED
from src.experiments import MAIN, build_main, unseen_experiment
from src.metrics import macro_f1

N_PARITY = 500


def fit_main(df):
    return build_main().fit(df["norm_text"], df["category"])


def quantize(W):
    """W: (classes, features). Returns int8 (features, classes) and per-feature float scales."""
    scale = np.abs(W).max(axis=0) / 127.0
    scale[scale == 0] = 1.0
    q = np.clip(np.rint(W / scale), -127, 127).astype(np.int8)
    return np.ascontiguousarray(q.T), scale


def quantized_proba(model, q, scale, texts):
    """Probabilities the browser will compute: same features, dequantized weights."""
    X = model[0].transform(texts)
    W = (q.T.astype(np.float64) * scale)
    logits = X @ W.T + model[-1].intercept_
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def export_model(model, out):
    union, clf = model[0], model[-1]
    (_, word), (_, char) = union.transformer_list
    q, scale = quantize(clf.coef_)
    vocab_list = lambda v: [t for t, _ in sorted(v.vocabulary_.items(), key=lambda kv: kv[1])]
    meta = {
        "config": MAIN,
        "classes": clf.classes_.tolist(),
        "intercept": np.round(clf.intercept_, 6).tolist(),
        "word": {"ngram_range": list(word.ngram_range), "vocab": vocab_list(word), "idf": np.round(word.idf_, 6).tolist()},
        "char": {"ngram_range": list(char.ngram_range), "vocab": vocab_list(char), "idf": np.round(char.idf_, 6).tolist()},
        "scale": [float(f"{s:.6g}") for s in scale],
        "n_features": int(q.shape[0]),
        "weights_layout": "int8, feature-major: byte j*n_classes + c is class c's weight for feature j, times scale[j]",
    }
    (out / "model.json").write_text(json.dumps(meta, separators=(",", ":")))
    (out / "weights.bin").write_bytes(q.tobytes())
    return q, scale


def round_list(xs, nd=4):
    return [round(float(x), nd) for x in xs]


def pick_examples(text, y, proba, classes):
    """Test messages for the demo buttons: confident and right, right but unsure, and wrong."""
    conf, pred = proba.max(axis=1), classes[proba.argmax(axis=1)]
    rng = np.random.default_rng(SEED)
    short = np.array([10 <= len(t) <= 90 for t in text])
    def take(mask, n):
        idx = np.flatnonzero(mask & short)
        return [{"text": str(text[i]), "label": str(y[i])} for i in rng.choice(idx, size=min(n, len(idx)), replace=False)]
    return {"clear": take((pred == y) & (conf > 0.9), 4),
            "ambiguous": take((pred == y) & (conf < 0.45), 4),
            "misclassified": take(pred != y, 4)}


def confusion(y, pred, order):
    """Counts[i][j] = messages of intent order[i] predicted as order[j]."""
    pos = {c: i for i, c in enumerate(order)}
    M = np.zeros((len(order), len(order)), dtype=int)
    np.add.at(M, ([pos[c] for c in y], [pos[c] for c in pred]), 1)
    return M.tolist()


def build_results(df):
    load = lambda name: json.loads((REPORTS / name).read_text())
    final = json.loads((REPORTS / "final" / "final_test.json").read_text())
    base, m1, m24, h2, h3, h4, h1 = (load(f) for f in ["baselines.json", "m1.json", "m2_m4.json", "h2.json",
                                                         "h3.json", "h4.json", "h1.json"])
    unseen, curve, e1, coarse = (load(f) for f in ["unseen.json", "learning_curve.json", "e1.json", "coarse_groups.json"])

    names = {"B0_majority": "Majority class", "B1_keyword_k10": "Keyword rule (k=10)", "B1_keyword_k20": "Keyword rule (k=20)",
             "M1_L2": "TF-IDF + logistic regression (L2)", "M1_L1": "TF-IDF + logistic regression (L1)",
             "M2_KNN": "KNN on TF-IDF", "M2_KNN_SVD": "KNN on SVD-300", "M3_SVD_LR": "SVD-300 + logistic regression",
             "M4_RF_SVD": "Random forest on SVD-300"}
    cv = {**{k: base[k] for k in ["B0_majority", "B1_keyword_k10", "B1_keyword_k20"]},
          **{k: m1["models"][k]["scores"] for k in ["M1_L2", "M1_L1"]},
          **{k: m24[k]["scores"] for k in ["M2_KNN", "M2_KNN_SVD", "M3_SVD_LR", "M4_RF_SVD"]}}
    scores = [{"id": k, "name": names[k],
               "cv": {"macro_f1": cv[k]["macro_f1"]["value"], "ci": cv[k]["macro_f1"]["ci95"], "accuracy": cv[k]["accuracy"]["value"]},
               "test": {"macro_f1": final["scores"][k]["macro_f1"]["value"], "ci": final["scores"][k]["macro_f1"]["ci95"],
                        "accuracy": final["scores"][k]["accuracy"]["value"]}} for k in names]

    oof = np.load(REPORTS / "oof" / "M1_L2.npz", allow_pickle=True)
    oof_classes, oof_pred = oof["classes"], oof["classes"][oof["proba"].argmax(axis=1)]
    test = np.load(REPORTS / "final" / "test_M1_L2.npz", allow_pickle=True)
    t_pred = test["classes"][test["proba"].argmax(axis=1)]

    print("U: recomputing seen/unseen OOF confidences (training data only, ~35 s)", flush=True)
    u = unseen_experiment(df)

    groups = json.loads((REPORTS / "h1.json").read_text())["a_within_group_share"]["groups_at_5"]
    grouped = [c for g in groups for c in g]
    order = grouped + sorted(set(oof_classes) - set(grouped))      # group members first, so blocks sit on the diagonal

    reliability = lambda bins: [{"conf": round(b["mean_confidence"], 4), "acc": round(b["accuracy"], 4), "n": b["count"]}
                                for b in bins if b["count"] > 0]
    return {
        "dataset": {"n_train": final["n_train"], "n_test": final["n_test"], "n_intents": 77,
                    "overlap": final["overlap"], "near_dup": json.loads((DATA_PROCESSED / "split_report.json").read_text())},
        "scores": scores,
        "headline": {"test": final["scores"]["M1_L2"]["macro_f1"], "vs_b1": final["M1_L2_vs_B1_keyword_k20_macro_f1"],
                     "cv_vs_b1": m1["models"]["M1_L2"]["vs_B1_keyword_k20_macro_f1"],
                     "without_overlap": final["scores_without_overlap"]["M1_L2"]["macro_f1"]["value"],
                     "seed_sd": m1["seed_spread"]["sd"], "near_dup_sensitivity": m1["sensitivity_near_dup_0.80"]["difference"]},
        "tuning": {"M1_L2": m1["models"]["M1_L2"]["selection"], "M2_KNN_grid": m24["M2_KNN"]["grid"]},
        "slider": {
            "cv": {"conf": round_list(oof["proba"].max(axis=1)), "correct": (oof_pred == df["category"].to_numpy()).astype(int).tolist()},
            "test": {"conf": round_list(test["proba"].max(axis=1)), "correct": (t_pred == test["y"]).astype(int).tolist()},
            "unseen": {"seen": round_list(u["_conf"]["seen"]), "unseen": round_list(u["_conf"]["unseen"])},
            "frozen_thresholds": [{"k": t["k"], "threshold": t["threshold"]} for t in h2["M1_L2"]["cost_thresholds"]],
        },
        "h2": {"cv": {"aurc": h2["M1_L2"]["aurc"], "aurc_random": h2["M1_L2"]["aurc_random"], "levels": h2["M1_L2"]["test"]["levels"],
                      "cost_thresholds": h2["M1_L2"]["cost_thresholds"], "random_band": h2["M1_L2"]["random_band"]},
               "test": {"aurc": final["H2"]["aurc"], "aurc_random": final["H2"]["aurc_random"], "levels": final["H2"]["test"]["levels"],
                        "frozen_thresholds": final["H2"]["frozen_thresholds_on_test"]}},
        "e1": {"cv": e1, "test": final["E1"]},
        "h3": {"cv": {k: {"name": names[k], "macro_f1": v["macro_f1"], "ece": v["ece"], "ece_ci": v["ece_ci95"],
                          "mean_confidence": v["mean_confidence"], "accuracy": v["accuracy"], "bins": reliability(v["reliability"])}
                      for k, v in h3["models"].items()},
               "test": {k: {"name": names[k], "macro_f1": v["macro_f1"], "ece": v["ece"], "ece_ci": v["ece_ci95"],
                            "mean_confidence": v["mean_confidence"], "accuracy": v["accuracy"], "bins": reliability(v["reliability"])}
                        for k, v in final["H3"].items()}},
        "h4": {"cv": [{"d": c["d"], "macro_f1": c["pooled_macro_f1"], "share": c["share_of_full_tfidf"],
                       "explained_variance": c["explained_variance"]} for c in h4["curve"]],
               "cv_full": h4["full_tfidf_macro_f1"],
               "test": [{"d": c["d"], "macro_f1": c["macro_f1"], "share": c["share_of_full_tfidf"]} for c in final["H4"]["curve"]],
               "test_full": final["H4"]["full_tfidf_macro_f1"]},
        "h1": {"edge": {k: v for k, v in h1["a_within_group_share"].items() if k != "groups_at_5" and k != "supported"},
               "groups": h1["a_within_group_share"]["groups_at_5"],
               "hand_labels": h1["b_hand_labels"], "cross_model": h1["c_cross_model_pairs"],
               "merge_cv": h1["merge"], "top3_cv": h1["top3_accuracy"], "coarse_cv": h1["coarse"],
               "test": final["H1"], "coarse_groups": coarse["groups"],
               "macro_f1_cv": macro_f1(df["category"].to_numpy(), oof_pred),
               "confusion": {"order": order, "cv": confusion(df["category"].to_numpy(), oof_pred, order),
                             "test": confusion(test["y"], t_pred, order)}},
        "unseen": {"cv": {k: unseen[k] for k in ["n_seen", "n_unseen", "auroc_seen_vs_unseen", "mean_confidence",
                                                 "at_cost_thresholds", "by_unseen_intent"]},
                   "test": final["U"],
                   "intents": json.loads((DATA_PROCESSED / "unseen_intents.json").read_text())},
        "learning_curve": {"cv": [{"fraction": c["fraction"], "n": c["mean_training_messages"], "mean": c["mean"], "sd": c["sd"]} for c in curve],
                           "test": [{"fraction": c["fraction"], "mean": c["mean"], "sd": c.get("sd", 0.0)} for c in final["learning_curve"]]},
        "examples": pick_examples(test["text"], test["y"], test["proba"], test["classes"]),
        "test_run": {"ran_at_utc": final["ran_at_utc"], "git_commit": final["git_commit"]},
    }


def main():
    out = DOCS_ASSETS
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PROCESSED / "train_split.csv")

    print("Fitting M1-L2 on all training data", flush=True)
    model = fit_main(df)
    q, scale = export_model(model, out)

    # Parity sample: Python's float probabilities and the dequantized ones the browser should reproduce.
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(df), size=N_PARITY, replace=False)
    texts = df["text"].to_numpy()[idx]                       # raw text: the browser normalizes it itself
    norm = df["norm_text"].to_numpy()[idx]
    p_float = model.predict_proba(norm)
    p_quant = quantized_proba(model, q, scale, norm)
    agree = float(np.mean(p_float.argmax(1) == p_quant.argmax(1)))
    print(f"int8 weights: top-1 agreement with float model {agree:.4f}; max |p diff| {np.abs(p_float - p_quant).max():.4f}")
    (out / "parity.json").write_text(json.dumps({
        "texts": texts.tolist(), "top1_float": model[-1].classes_[p_float.argmax(1)].tolist(),
        "p_quant": [[round(float(x), 6) for x in row] for row in p_quant],
        "quantization_top1_agreement": agree}, separators=(",", ":")))

    results = build_results(df)
    (out / "results.json").write_text(json.dumps(results, separators=(",", ":")))
    for f in ["model.json", "weights.bin", "results.json", "parity.json"]:
        print(f"  {f}: {(out / f).stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
