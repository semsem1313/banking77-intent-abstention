"""Tests for the error predictor's features and out-of-fold training."""
import numpy as np

from src.error_predictor import confidence_features, oof_error_scores


def test_confidence_features():
    proba = np.array([[0.7, 0.2, 0.1], [1 / 3, 1 / 3, 1 / 3]])
    X = confidence_features(proba, ["short", "a longer message"])
    assert np.allclose(X["max_prob"], [0.7, 1 / 3])
    assert np.allclose(X["margin"], [0.5, 0.0])
    assert X["entropy"][1] > X["entropy"][0]          # a uniform guess is the most uncertain
    assert list(X["length"]) == [5, 16]


def test_oof_error_scores_learn_that_low_confidence_means_wrong():
    rng = np.random.default_rng(0)
    conf = rng.random(1000)
    wrong = conf < 0.3
    X = np.column_stack([conf, rng.random(1000), rng.random(1000), rng.random(1000)])
    scores = oof_error_scores(X, wrong, np.arange(1000) % 5)
    assert scores[wrong].mean() > scores[~wrong].mean() + 0.3
