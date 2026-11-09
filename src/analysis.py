"""Hypothesis analyses from out-of-fold (OOF) predictions. Never reads test.csv.

H2: confidence-based abstention vs random abstention (risk-coverage), plus
    cost-based handoff thresholds for each cost ratio k.
H3: calibration (ECE, reliability diagram) vs accuracy across models.

Decision rules are in PREREGISTRATION.md. H1, H4, and U are added as their models exist.

Usage: python -m src.analysis  (writes reports/h2.json, reports/h3.json, reports/figures/*.png)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.baselines import random_abstention_error
from src.config import COST_RATIOS, COVERAGE_GRID, DATA_PROCESSED, ECE_BINS, N_BOOTSTRAP, REPORTS, SEED
from src.metrics import macro_f1

MODELS = {"M1_L2": "M1-L2 (main)", "M1_L1": "M1-L1"}
CURVE_COVERAGES = np.round(np.arange(0.05, 1.0001, 0.01), 2)


def load_oof(name):
    d = np.load(REPORTS / "oof" / f"{name}.npz", allow_pickle=True)
    return d["proba"].astype(float), d["classes"]


# ---------- H2: abstention ----------

def selective_error(conf, correct, coverage):
    """Answer the `coverage` fraction of messages with the highest confidence; error rate on those.

    Ties in confidence are broken by original order (a stable sort), so the result is deterministic.
    """
    m = max(1, int(round(coverage * len(conf))))
    answered = np.argsort(-conf, kind="stable")[:m]
    return float(1 - correct[answered].mean())


def h2_test(conf, correct, coverages=COVERAGE_GRID, n_resamples=N_BOOTSTRAP, seed=SEED):
    """Pre-registered H2 test: at each coverage, the difference (confidence error - random error).

    Random abstention's expected error is the overall error rate, so each bootstrap resample
    compares the confidence rule against the overall error of that same resample (paired).
    """
    rng = np.random.default_rng(seed)
    n = len(conf)
    point = {c: selective_error(conf, correct, c) - (1 - correct.mean()) for c in coverages}
    diffs = {c: [] for c in coverages}
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        cf, cr = conf[idx], correct[idx]
        base = 1 - cr.mean()
        for c in coverages:
            diffs[c].append(selective_error(cf, cr, c) - base)
    levels = {}
    for c in coverages:
        lo, hi = np.quantile(diffs[c], [0.025, 0.975])
        levels[str(c)] = {"selective_error": point[c] + (1 - correct.mean()), "random_error": float(1 - correct.mean()),
                          "difference": point[c], "ci95": [float(lo), float(hi)], "ci_excludes_zero_below": bool(hi < 0)}
    supported = all(v["ci_excludes_zero_below"] for v in levels.values())
    return {"levels": levels, "supported": supported}


def cost_threshold(conf, correct, k):
    """Choose the confidence threshold t that minimizes cost per message.

    Answer when conf >= t, otherwise hand off. Cost = k for a wrong answer, 1 for a handoff,
    0 for a right answer. Candidate thresholds are the observed confidence values (plus "answer
    everything"), so the search is exact.
    """
    order = np.argsort(-conf, kind="stable")
    wrong_sorted = (~correct[order]).astype(float)
    n = len(conf)
    answered = np.arange(0, n + 1)                       # answer the top-m messages, m = 0..n
    wrong_answered = np.concatenate([[0], np.cumsum(wrong_sorted)])
    cost = (k * wrong_answered + (n - answered)) / n
    # Only cut between distinct confidence values: a threshold can't split tied messages.
    conf_sorted = conf[order]
    valid = np.ones(n + 1, dtype=bool)
    valid[1:n] = conf_sorted[:-1] != conf_sorted[1:]
    m = int(np.flatnonzero(valid)[np.argmin(cost[valid])])
    threshold = float(conf_sorted[m - 1]) if m > 0 else 1.0
    return {
        "k": k, "threshold": threshold, "coverage": m / n,
        "selective_accuracy": float(1 - wrong_answered[m] / m) if m else None,
        "cost_per_message": float(cost[m]),
        "cost_answer_all": float(k * wrong_sorted.sum() / n),
        "cost_handoff_all": 1.0,
    }


def h2(name):
    proba, classes = load_oof(name)
    y = pd.read_csv(DATA_PROCESSED / "train_split.csv")["category"].to_numpy()
    conf = proba.max(axis=1)
    correct = classes[proba.argmax(axis=1)] == y
    curve = [{"coverage": float(c), "selective_error": selective_error(conf, correct, c)} for c in CURVE_COVERAGES]
    random_band = random_abstention_error(correct, list(CURVE_COVERAGES), n_draws=200)
    return {
        "model": name,
        "overall_error": float(1 - correct.mean()),
        "mean_confidence_correct": float(conf[correct].mean()),
        "mean_confidence_wrong": float(conf[~correct].mean()),
        "aurc": float(np.mean([p["selective_error"] for p in curve])),   # area under the risk-coverage curve
        "aurc_random": float(1 - correct.mean()),
        "test": h2_test(conf, correct),
        "cost_thresholds": [cost_threshold(conf, correct, k) for k in COST_RATIOS],
        "curve": curve,
        "random_band": {str(c): v for c, v in random_band.items()},
    }


# ---------- H3: calibration ----------

def reliability_bins(conf, correct, n_bins=ECE_BINS):
    """Equal-width bins on top-label confidence: count, mean confidence, accuracy per bin."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        mask = idx == b
        bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "count": int(mask.sum()),
                     "mean_confidence": float(conf[mask].mean()) if mask.any() else None,
                     "accuracy": float(correct[mask].mean()) if mask.any() else None})
    return bins


def ece(conf, correct, n_bins=ECE_BINS):
    """Expected calibration error: the count-weighted mean |accuracy - confidence| over bins."""
    n = len(conf)
    return float(sum(b["count"] / n * abs(b["accuracy"] - b["mean_confidence"])
                     for b in reliability_bins(conf, correct, n_bins) if b["count"]))


def ece_ci(conf, correct, n_resamples=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(conf)
    stats = [ece(conf[idx], correct[idx]) for idx in (rng.integers(0, n, n) for _ in range(n_resamples))]
    return [float(q) for q in np.quantile(stats, [0.025, 0.975])]


def h3(names):
    y = pd.read_csv(DATA_PROCESSED / "train_split.csv")["category"].to_numpy()
    rows = {}
    for name in names:
        proba, classes = load_oof(name)
        pred = classes[proba.argmax(axis=1)]
        conf, correct = proba.max(axis=1), pred == y
        rows[name] = {"macro_f1": macro_f1(y, pred), "ece": ece(conf, correct), "ece_ci95": ece_ci(conf, correct),
                      "mean_confidence": float(conf.mean()), "accuracy": float(correct.mean()),
                      "reliability": reliability_bins(conf, correct)}
    most_accurate = max(rows, key=lambda m: rows[m]["macro_f1"])
    best_calibrated = min(rows, key=lambda m: rows[m]["ece"])
    a, b = rows[most_accurate]["ece_ci95"], rows[best_calibrated]["ece_ci95"]
    cis_separate = most_accurate != best_calibrated and (a[0] > b[1] or b[0] > a[1])
    return {
        "models": rows,
        "most_accurate": most_accurate,
        "best_calibrated": best_calibrated,
        "supported": bool(cis_separate),
        "provisional": "Only the models trained so far are compared; the verdict is final once all models exist.",
    }


# ---------- Figures ----------

def plot_h2(results, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for (name, r), color in zip(results.items(), ["tab:blue", "tab:orange"]):
        cov = [p["coverage"] for p in r["curve"]]
        ax.plot(cov, [p["selective_error"] for p in r["curve"]], color=color, label=f"{MODELS[name]}: abstain by confidence")
        band = r["random_band"]
        ax.fill_between(cov, [band[str(c)]["ci95"][0] for c in cov], [band[str(c)]["ci95"][1] for c in cov], color=color, alpha=0.15)
        ax.plot(cov, [band[str(c)]["mean"] for c in cov], "--", color=color, label=f"{MODELS[name]}: random abstention (B2)")
    ax.set(xlabel="Coverage (share of messages answered automatically)", ylabel="Error rate on answered messages",
           title="H2: risk-coverage (OOF, 9,999 messages)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_h3(result, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="grey", lw=1, label="perfect calibration")
    for name, r in result["models"].items():
        pts = [(b["mean_confidence"], b["accuracy"]) for b in r["reliability"] if b["count"] >= 20]
        ax.plot(*zip(*pts), "o-", label=f"{MODELS[name]} (ECE {r['ece']:.3f})")
    ax.set(xlabel="Mean predicted confidence (bin)", ylabel="Accuracy (bin)", title="H3: reliability (OOF; bins with >= 20 messages)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    h2_results = {name: h2(name) for name in MODELS}
    (REPORTS / "h2.json").write_text(json.dumps(h2_results, indent=2))
    plot_h2(h2_results, REPORTS / "figures" / "h2_risk_coverage.png")
    for name, r in h2_results.items():
        print(f"H2 {name}: supported={r['test']['supported']}  AURC {r['aurc']:.4f} vs random {r['aurc_random']:.4f}")
        for t in r["cost_thresholds"]:
            print(f"   k={t['k']:>2}: threshold {t['threshold']:.3f}, coverage {t['coverage']:.3f}, "
                  f"selective acc {t['selective_accuracy']:.3f}, cost {t['cost_per_message']:.4f} "
                  f"(answer all {t['cost_answer_all']:.4f}, hand off all 1.0)")

    h3_result = h3(list(MODELS))
    (REPORTS / "h3.json").write_text(json.dumps(h3_result, indent=2))
    plot_h3(h3_result, REPORTS / "figures" / "h3_reliability.png")
    for name, r in h3_result["models"].items():
        print(f"H3 {name}: macro-F1 {r['macro_f1']:.4f}  ECE {r['ece']:.4f} {r['ece_ci95']}  mean conf {r['mean_confidence']:.3f} vs acc {r['accuracy']:.3f}")
    print(f"H3: most accurate {h3_result['most_accurate']}, best calibrated {h3_result['best_calibrated']}, supported={h3_result['supported']}")


if __name__ == "__main__":
    main()
