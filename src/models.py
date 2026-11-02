"""Model definitions and their hyperparameter grids.

Every model is a full sklearn Pipeline (features + classifier), so the TF-IDF
vocabulary and IDF weights are refit inside each training fold. That's how
preprocessing leakage is prevented.

M1 (main model): TF-IDF + logistic regression.
  - L2: multinomial (softmax) logistic regression, lbfgs solver.
  - L1: one-vs-rest L1 logistic regression, liblinear solver. A multinomial L1 fit
    with saga took over 6 minutes per fold without converging, so one-vs-rest is used.
Later models (M2-M5, E1) are added here as the course reaches them.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline, make_union

from src.features import char_tfidf, word_tfidf

FEATURE_SETS = {
    "word": lambda: word_tfidf(ngram_range=(1, 2)),
    "word+char": lambda: make_union(word_tfidf(ngram_range=(1, 2)), char_tfidf(ngram_range=(3, 5))),
}

# Grids are ordered from simplest (strongest regularization, fewest features) to most complex.
# The one-standard-error rule in tune.py relies on this order.
M1_GRID = {
    "M1_L2": [{"features": f, "C": C} for f in ("word", "word+char") for C in (1, 3, 10, 30, 100)],
    "M1_L1": [{"features": "word+char", "C": C} for C in (1, 3, 10, 30)],
}


def build_m1(penalty, features, C):
    feats = FEATURE_SETS[features]()
    if penalty == "L2":
        clf = LogisticRegression(C=C, max_iter=3000)
    elif penalty == "L1":
        clf = OneVsRestClassifier(LogisticRegression(C=C, l1_ratio=1.0, solver="liblinear", max_iter=2000))
    else:
        raise ValueError(penalty)
    return make_pipeline(feats, clf)


def config_name(family, cfg):
    return f"{family}[{cfg['features']},C={cfg['C']}]"
