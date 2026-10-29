"""Unit tests for the baselines on tiny hand-made data."""
import numpy as np
import pandas as pd

from src.baselines import KeywordRule, MajorityClass, oof_predict, random_abstention_error

TEXTS = ["my card is lost", "lost my card again", "card lost today",
         "top up failed", "my top up failed", "top up failed again",
         "hello there"]
LABELS = ["lost", "lost", "lost", "topup", "topup", "topup", "lost"]


def test_majority_class_predicts_most_frequent():
    model = MajorityClass().fit(TEXTS, LABELS)
    assert set(model.predict(["anything", "else"])) == {"lost"}


def test_majority_class_breaks_ties_alphabetically():
    assert MajorityClass().fit(["a", "b"], ["zeta", "alpha"]).label_ == "alpha"


def test_keyword_rule_uses_distinctive_words():
    model = KeywordRule(k=2, min_count=2).fit(TEXTS, LABELS)
    assert model.predict(["why has my top up failed"])[0] == "topup"
    assert model.predict(["i lost it"])[0] == "lost"


def test_keyword_rule_falls_back_when_nothing_matches():
    model = KeywordRule(k=2, min_count=2).fit(TEXTS, LABELS)
    assert model.predict(["completely unrelated words"])[0] == "lost"  # the majority class


def test_oof_never_trains_on_the_predicted_fold():
    # A model that memorizes its training texts. If the held-out fold leaked into training,
    # every prediction would be "seen".
    class Memorizer:
        def fit(self, texts, labels):
            self.seen = set(texts)
            return self

        def predict(self, texts):
            return np.array(["seen" if t in self.seen else "unseen" for t in texts], dtype=object)

    df = pd.DataFrame({"norm_text": [f"m{i}" for i in range(10)], "category": ["x"] * 10, "fold": [i % 5 for i in range(10)]})
    assert set(oof_predict(Memorizer, df, "fold")) == {"unseen"}


def test_random_abstention_matches_overall_error():
    correct = np.array([True] * 80 + [False] * 20)
    out = random_abstention_error(correct, [0.5, 0.9], n_draws=2000)
    for cov in (0.5, 0.9):
        assert abs(out[cov]["mean"] - 0.20) < 0.01
