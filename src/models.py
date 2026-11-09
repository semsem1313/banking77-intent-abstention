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
from sklearn.neighbors import KNeighborsClassifier
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


# M2: cosine KNN on the same word+char TF-IDF. Larger k = smoother (simpler) decision boundary,
# so the grid runs from large k (simplest) to small k for the one-SE rule.
M2_GRID = [{"k": k, "weights": w} for k in (40, 20, 10, 5, 3, 1) for w in ("uniform", "distance")]


def build_knn(k, weights):
    return make_pipeline(FEATURE_SETS["word+char"](),
                         KNeighborsClassifier(n_neighbors=k, weights=weights, metric="cosine", algorithm="brute"))


# M3 / M4 / H4: TruncatedSVD of word+char TF-IDF, fit inside each training fold (see tune_more.py).
SVD_MAX_DIM = 1000
SVD_DIMS = [25, 50, 100, 200, 300, 500, 1000]
SVD_LR_C = [1, 10, 100]          # simplest (strongest regularization) first
SVD_MAIN_DIM = 300               # the pre-registered H4 boundary; M3 and M4 are reported at this size
RF_PARAMS = {"n_estimators": 300, "max_features": "sqrt", "min_samples_leaf": 1}
