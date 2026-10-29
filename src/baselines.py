"""Baselines B0 (majority class), B1 (keyword rule), B2 (random abstention).

B0 and B1 are scored with out-of-fold (OOF) predictions on the main folds (fold_<SEED>).
Each fold is predicted by a baseline fit only on the other four folds.
B2 is a reference curve for H2: it needs a model's correct/wrong flags, so the
analysis step applies it to the main model later.

Usage: python -m src.baselines   (writes reports/baselines.json and reports/oof/baselines.csv)
"""
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.config import DATA_PROCESSED, REPORTS, SEED
from src.features import TOKEN_PATTERN
from src.metrics import score

KEYWORD_K = 10          # keywords per intent, fixed in advance (primary)
KEYWORD_K_SENSITIVITY = [5, 20]
KEYWORD_MIN_COUNT = 3   # a word must appear in at least 3 messages of an intent to be one of its keywords


class MajorityClass:
    """B0: always predict the most frequent intent in the training data."""

    def fit(self, texts, labels):
        counts = pd.Series(labels).value_counts()
        # Ties are broken alphabetically so the result is deterministic.
        self.label_ = sorted(counts[counts == counts.max()].index)[0]
        return self

    def predict(self, texts):
        return np.full(len(texts), self.label_, dtype=object)


class KeywordRule:
    """B1: a transparent keyword rule, learned from the training folds.

    Fit:     for each intent, pick the k words with the highest precision
             P(intent | word appears) among words seen in >= min_count of its messages.
    Predict: each intent scores the sum of the precisions of its keywords present in
             the message. The highest score wins. If no keyword matches, fall back to B0.
    """

    def __init__(self, k=KEYWORD_K, min_count=KEYWORD_MIN_COUNT):
        self.k = k
        self.min_count = min_count

    def fit(self, texts, labels):
        labels = np.asarray(labels)
        self.vectorizer_ = CountVectorizer(lowercase=False, token_pattern=TOKEN_PATTERN, binary=True)
        X = self.vectorizer_.fit_transform(texts)               # messages x words, 1 if word present
        self.classes_ = np.array(sorted(set(labels)))
        Y = (labels[:, None] == self.classes_[None, :]).astype(float)  # messages x intents
        in_class = np.asarray((X.T @ Y))                         # words x intents: n(word, intent)
        overall = np.asarray(X.sum(axis=0)).ravel()[:, None]     # words x 1: n(word)
        precision = in_class / overall

        weights = np.zeros_like(precision)
        vocab = np.array(self.vectorizer_.get_feature_names_out())
        self.keywords_ = {}
        for c, intent in enumerate(self.classes_):
            eligible = np.flatnonzero(in_class[:, c] >= self.min_count)
            # Sort by precision, then by count (more evidence first), then alphabetically.
            order = sorted(eligible, key=lambda w: (-precision[w, c], -in_class[w, c], vocab[w]))[: self.k]
            weights[order, c] = precision[order, c]
            self.keywords_[intent] = [(vocab[w], round(float(precision[w, c]), 3)) for w in order]
        self.weights_ = weights
        self.fallback_ = MajorityClass().fit(texts, labels).label_
        return self

    def predict(self, texts):
        scores = self.vectorizer_.transform(texts) @ self.weights_  # messages x intents
        pred = self.classes_[np.argmax(scores, axis=1)].astype(object)
        pred[scores.max(axis=1) == 0] = self.fallback_
        return pred


def random_abstention_error(correct, coverages, n_draws=1000, seed=SEED):
    """B2: error rate on the answered messages when a random fraction `coverage` is answered.

    In expectation this equals the overall error rate at every coverage level. The draws
    give its spread, so the H2 comparison is against a distribution, not a single number.
    """
    correct = np.asarray(correct, dtype=bool)
    rng = np.random.default_rng(seed)
    n = len(correct)
    out = {}
    for cov in coverages:
        m = int(round(cov * n))
        errors = [1 - correct[rng.choice(n, m, replace=False)].mean() for _ in range(n_draws)]
        out[cov] = {"mean": float(np.mean(errors)), "ci95": [float(q) for q in np.quantile(errors, [0.025, 0.975])]}
    return out


def oof_predict(make_model, df, fold_col):
    """Fit on four folds, predict the fifth, repeat. Returns one prediction per row."""
    pred = np.empty(len(df), dtype=object)
    for k in sorted(df[fold_col].unique()):
        train, held_out = df[fold_col] != k, df[fold_col] == k
        model = make_model().fit(df.loc[train, "norm_text"], df.loc[train, "category"])
        pred[held_out.to_numpy()] = model.predict(df.loc[held_out, "norm_text"])
    return pred


def main():
    df = pd.read_csv(DATA_PROCESSED / "train_split.csv")
    fold_col = f"fold_{SEED}"
    y, folds = df["category"], df[fold_col]

    runs = {"B0_majority": MajorityClass, f"B1_keyword_k{KEYWORD_K}": lambda: KeywordRule(KEYWORD_K)}
    for k in KEYWORD_K_SENSITIVITY:
        runs[f"B1_keyword_k{k}"] = lambda k=k: KeywordRule(k)

    oof = pd.DataFrame({"category": y, fold_col: folds})
    results = {}
    for name, make_model in runs.items():
        oof[name] = oof_predict(make_model, df, fold_col)
        results[name] = score(y, oof[name], folds)
        print(f"{name:18s} accuracy {results[name]['accuracy']['value']:.3f}  macro-F1 {results[name]['macro_f1']['value']:.3f}")

    # How often B1 had no matching keyword and fell back to the majority class.
    full = KeywordRule(KEYWORD_K).fit(df["norm_text"], y)
    no_match = (full.vectorizer_.transform(df["norm_text"]) @ full.weights_).max(axis=1) == 0
    results[f"B1_keyword_k{KEYWORD_K}"]["share_no_keyword_match_in_sample"] = float(np.mean(no_match))
    results["keywords_fit_on_all_training_data_for_display"] = full.keywords_

    (REPORTS / "oof").mkdir(parents=True, exist_ok=True)
    oof.to_csv(REPORTS / "oof" / "baselines.csv", index=False)
    (REPORTS / "baselines.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
