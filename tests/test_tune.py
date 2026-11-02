"""Tests for model selection and the paired bootstrap."""
import numpy as np

from src.metrics import accuracy, paired_bootstrap_diff
from src.models import M1_GRID, build_m1
from src.tune import one_se_choice


def _row(name, mean, sd=0.01):
    return {"name": name, "macro_f1": {"mean": mean, "sd": sd, "per_fold": [0] * 5}}


def test_one_se_picks_simplest_within_one_standard_error():
    # SE of the best = 0.01 / sqrt(5) ≈ 0.0045, so anything >= 0.8955 qualifies.
    rows = [_row("simple", 0.880), _row("medium", 0.897), _row("complex", 0.900)]
    chosen, detail = one_se_choice(rows)
    assert chosen["name"] == "medium"
    assert detail["best"] == "complex"


def test_one_se_returns_best_when_nothing_else_is_close():
    rows = [_row("simple", 0.80), _row("complex", 0.90)]
    assert one_se_choice(rows)[0]["name"] == "complex"


def test_grids_go_from_strong_to_weak_regularization():
    for configs in M1_GRID.values():
        for features in {c["features"] for c in configs}:
            Cs = [c["C"] for c in configs if c["features"] == features]
            assert Cs == sorted(Cs)


def test_paired_bootstrap_identical_models_have_zero_difference():
    y = np.array(list("aabbccdd") * 10)
    out = paired_bootstrap_diff(y, y, y, accuracy, n_resamples=200)
    assert out["diff"] == 0 and out["ci95"] == (0.0, 0.0)


def test_paired_bootstrap_detects_a_clearly_better_model():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 500)
    good = np.where(rng.random(500) < 0.9, y, (y + 1) % 3)
    bad = np.where(rng.random(500) < 0.5, y, (y + 1) % 3)
    out = paired_bootstrap_diff(y, good, bad, accuracy, n_resamples=500)
    assert out["ci95"][0] > 0


def test_pipeline_learns_tfidf_inside_fit():
    model = build_m1("L2", "word", 1.0).fit(["card lost", "top up failed", "lost card", "failed top up"], ["a", "b", "a", "b"])
    vocab = model[0].vocabulary_
    assert "card" in vocab and "unseen" not in vocab
    assert model.predict(["my card is lost"])[0] == "a"
