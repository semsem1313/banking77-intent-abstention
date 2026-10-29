"""Checks on the pre-registered split (built in memory, so these don't depend on files in data/processed)."""
import pandas as pd
import pytest

from src.config import N_FOLDS, N_UNSEEN_INTENTS, SEED, SEEDS
from src.split import build_split


@pytest.fixture(scope="module")
def split():
    return build_split()


def test_no_exact_duplicates_remain(split):
    df, _ = split
    assert not df["norm_text"].duplicated().any()


@pytest.mark.parametrize("group_col, fold_col", [("group", f"fold_{SEED}"), ("group_sens", "fold_sens")])
def test_groups_never_cross_folds(split, group_col, fold_col):
    df, _ = split
    assert (df.groupby(group_col)[fold_col].nunique() == 1).all()


def test_every_intent_in_every_fold(split):
    df, _ = split
    for seed in SEEDS:
        counts = pd.crosstab(df["category"], df[f"fold_{seed}"])
        assert counts.shape == (77, N_FOLDS)
        assert (counts > 0).all().all()


def test_sensitivity_groups_are_coarser(split):
    df, _ = split
    # Every 0.90 group must sit inside a single 0.80 group.
    assert (df.groupby("group")["group_sens"].nunique() == 1).all()


def test_unseen_intents(split):
    df, report = split
    assert len(report["unseen_intents"]) == N_UNSEEN_INTENTS
    assert set(df.loc[df["is_unseen"], "category"]) == set(report["unseen_intents"])


def test_split_is_reproducible(split):
    df, _ = split
    df2, _ = build_split()
    pd.testing.assert_frame_equal(df, df2)
