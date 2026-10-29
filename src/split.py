"""Build the pre-registered training split. Never reads test.csv.

Steps (PREREGISTRATION.md, "Split and data rules"):
1. Normalize text and remove exact duplicates (keep the first copy).
2. Link training messages whose word 1-2 gram TF-IDF cosine is >= NEAR_DUP_COSINE.
   Connected components become groups.
3. Assign StratifiedGroupKFold folds (stratified by intent, grouped by component),
   once per seed in SEEDS. fold_<SEED> is the main assignment.
4. Sensitivity check (amendment 2026-10-29): groups and folds at the stricter
   SENSITIVITY_NEAR_DUP_COSINE, in group_sens and fold_sens.
5. Draw the unseen intents with UNSEEN_SEED.

Outputs in data/processed/:
  train_split.csv    one row per kept message, with group, folds, and an is_unseen flag
  unseen_intents.json
  split_report.json  counts and checks, used by the site's Data section

Usage: python -m src.split
"""
import json

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
from sklearn.model_selection import StratifiedGroupKFold

from src.config import (
    DATA_PROCESSED, N_FOLDS, N_UNSEEN_INTENTS, NEAR_DUP_COSINE, SEED, SEEDS,
    SENSITIVITY_NEAR_DUP_COSINE, TRAIN_CSV, UNSEEN_SEED,
)
from src.features import normalize_text, word_tfidf

CHUNK = 2000


def near_duplicate_pairs(texts, threshold):
    """Return (i, j, cosine) for all pairs i < j with cosine >= threshold."""
    X = word_tfidf(ngram_range=(1, 2)).fit_transform(texts)  # rows are L2-normalized
    rows, cols, sims = [], [], []
    for start in range(0, X.shape[0], CHUNK):
        block = (X[start:start + CHUNK] @ X.T).tocoo()
        i = block.row + start
        keep = (block.data >= threshold) & (i < block.col)
        rows.append(i[keep])
        cols.append(block.col[keep])
        sims.append(block.data[keep])
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(sims)


def group_ids(n, rows, cols):
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels


def assign_folds(y, groups, seed):
    folds = np.full(len(y), -1)
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for k, (_, val_idx) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
        folds[val_idx] = k
    return folds


def draw_unseen_intents(intents):
    rng = np.random.default_rng(UNSEEN_SEED)
    return sorted(rng.choice(sorted(intents), size=N_UNSEEN_INTENTS, replace=False).tolist())


def build_split():
    raw = pd.read_csv(TRAIN_CSV)
    df = raw.assign(norm_text=raw["text"].map(normalize_text))

    dup_mask = df["norm_text"].duplicated(keep="first")
    conflicts = int((df.groupby("norm_text")["category"].nunique() > 1).sum())
    df = df[~dup_mask].reset_index(drop=True)

    all_rows, all_cols, all_sims = near_duplicate_pairs(df["norm_text"], SENSITIVITY_NEAR_DUP_COSINE)
    primary = all_sims >= NEAR_DUP_COSINE
    rows, cols, sims = all_rows[primary], all_cols[primary], all_sims[primary]
    df["group"] = group_ids(len(df), rows, cols)
    df["group_sens"] = group_ids(len(df), all_rows, all_cols)

    labels = df["category"].values
    cross = labels[rows] != labels[cols]
    cross_label_examples = [
        {"cosine": round(float(s), 3), "text_a": df["norm_text"][i], "label_a": labels[i],
         "text_b": df["norm_text"][j], "label_b": labels[j]}
        for i, j, s in zip(rows[cross], cols[cross], sims[cross])
    ]

    for seed in SEEDS:
        df[f"fold_{seed}"] = assign_folds(df["category"], df["group"], seed)
    df["fold_sens"] = assign_folds(df["category"], df["group_sens"], SEED)

    unseen = draw_unseen_intents(df["category"].unique())
    df["is_unseen"] = df["category"].isin(unseen)

    group_sizes = df["group"].value_counts()
    main_fold = df[f"fold_{SEED}"]
    per_fold_class_min = int(df.groupby([main_fold, "category"]).size().unstack(fill_value=0).min().min())
    report = {
        "raw_rows": len(raw),
        "exact_duplicates_removed": int(dup_mask.sum()),
        "duplicate_texts_with_conflicting_labels": conflicts,
        "rows_after_dedup": len(df),
        "n_intents": int(df["category"].nunique()),
        "examples_per_intent": {
            "min": int(df["category"].value_counts().min()),
            "median": int(df["category"].value_counts().median()),
            "max": int(df["category"].value_counts().max()),
        },
        "near_duplicate_threshold": NEAR_DUP_COSINE,
        "near_duplicate_pairs": len(rows),
        "near_duplicate_pairs_with_different_labels": cross_label_examples,
        "n_groups": int(group_sizes.size),
        "rows_in_multi_message_groups": int(group_sizes[group_sizes > 1].sum()),
        "largest_group": int(group_sizes.max()),
        "sensitivity": {
            "threshold": SENSITIVITY_NEAR_DUP_COSINE,
            "near_duplicate_pairs": len(all_rows),
            "near_duplicate_pairs_with_different_labels": int((labels[all_rows] != labels[all_cols]).sum()),
            "n_groups": int(df["group_sens"].nunique()),
            "largest_group": int(df["group_sens"].value_counts().max()),
        },
        "fold_sizes": {str(k): int(v) for k, v in main_fold.value_counts().sort_index().items()},
        "min_examples_of_any_intent_in_any_fold": per_fold_class_min,
        "unseen_intents": unseen,
        "unseen_rows": int(df["is_unseen"].sum()),
    }
    return df, report


def main():
    df, report = build_split()
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PROCESSED / "train_split.csv", index=False)
    (DATA_PROCESSED / "unseen_intents.json").write_text(json.dumps(report["unseen_intents"], indent=2))
    (DATA_PROCESSED / "split_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
