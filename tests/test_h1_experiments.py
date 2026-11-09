"""Tests for H1 grouping logic and the nested learning-curve subsets."""
import numpy as np

from src.experiments import nested_order, subset
from src.h1 import COARSE_DRAFT, chance_share, confusable_groups, merge_labels, pair_counts, share_within_groups

CLASSES = np.array(["a", "b", "c", "d"])


def test_pair_counts_are_symmetric():
    y = np.array(["a", "a", "b", "c"])
    pred = np.array(["b", "a", "a", "d"])
    S = pair_counts(y, pred, CLASSES)
    assert S[0, 1] == S[1, 0] == 2 and S[2, 3] == 1 and (S == S.T).all()


def test_groups_follow_the_edge_threshold():
    S = np.zeros((4, 4), dtype=int)
    S[0, 1] = S[1, 0] = 6
    S[2, 3] = S[3, 2] = 2
    assert confusable_groups(S, CLASSES, 5) == [["a", "b"]]
    assert sorted(confusable_groups(S, CLASSES, 2)) == [["a", "b"], ["c", "d"]]


def test_share_and_chance_within_groups():
    y = np.array(["a", "a", "c"])
    pred = np.array(["b", "c", "d"])           # 3 errors, 1 inside group {a, b}
    groups = [["a", "b"]]
    assert share_within_groups(y, pred, groups)[0] == 1 / 3
    # Errors on "a" (twice) land in its 2-intent group by chance with probability 1/3; "c" has none.
    assert abs(chance_share(y, pred, groups, 4) - (2 * (1 / 3)) / 3) < 1e-12


def test_merge_labels_only_touches_group_members():
    assert list(merge_labels(np.array(["a", "b", "c"]), [["a", "b"]])) == ["a+b", "a+b", "c"]


def test_coarse_draft_covers_each_intent_once():
    members = sum(COARSE_DRAFT.values(), [])
    assert len(members) == 77 and len(set(members)) == 77


def test_learning_curve_subsets_are_nested_stratified_and_nonempty():
    labels = np.array(["x"] * 40 + ["y"] * 10)
    order = nested_order(labels, np.random.default_rng(0))
    small, large = subset(order, 0.05), subset(order, 0.5)
    assert set(small) <= set(large)
    assert {"x", "y"} <= set(labels[small])          # every class appears, even at 5%
    assert len(subset(order, 1.0)) == 50
