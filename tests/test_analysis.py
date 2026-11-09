"""Tests for the H2 and H3 analysis functions on small hand-made data."""
import numpy as np

from src.analysis import cost_threshold, ece, h2_test, reliability_bins, selective_error


def test_selective_error_answers_most_confident_first():
    conf = np.array([0.9, 0.8, 0.3, 0.2])
    correct = np.array([True, True, False, False])
    assert selective_error(conf, correct, 0.5) == 0.0
    assert selective_error(conf, correct, 1.0) == 0.5


def test_h2_supported_when_confidence_separates_errors():
    rng = np.random.default_rng(0)
    correct = rng.random(2000) < 0.8
    conf = np.where(correct, rng.uniform(0.6, 1.0, 2000), rng.uniform(0.0, 0.5, 2000))
    assert h2_test(conf, correct, coverages=[0.5, 0.7], n_resamples=100)["supported"]


def test_h2_not_supported_when_confidence_is_random():
    rng = np.random.default_rng(1)
    correct = rng.random(2000) < 0.8
    conf = rng.random(2000)
    assert not h2_test(conf, correct, coverages=[0.5, 0.7], n_resamples=100)["supported"]


def test_cost_threshold_hands_off_wrong_low_confidence_messages():
    conf = np.array([0.95, 0.9, 0.85, 0.2, 0.1])
    correct = np.array([True, True, True, False, False])
    out = cost_threshold(conf, correct, k=5)
    assert out["coverage"] == 0.6 and out["threshold"] == 0.85
    assert out["cost_per_message"] == 2 / 5  # two handoffs, no wrong answers


def test_cost_threshold_never_splits_tied_confidences():
    conf = np.array([0.5, 0.5, 0.5, 0.5])
    correct = np.array([True, False, True, False])
    assert cost_threshold(conf, correct, k=2)["coverage"] in (0.0, 1.0)


def test_ece_zero_for_perfect_calibration_and_large_for_overconfidence():
    conf = np.full(1000, 0.7)
    correct = np.arange(1000) < 700
    assert ece(conf, correct) < 1e-9
    assert abs(ece(np.full(1000, 0.95), correct) - 0.25) < 1e-9


def test_reliability_bins_cover_every_message():
    conf = np.random.default_rng(2).random(500)
    assert sum(b["count"] for b in reliability_bins(conf, conf > 0.5)) == 500
