"""Train/test split and ranking metrics used by scripts/evaluate.py."""

from __future__ import annotations

import numpy as np
import pandas as pd


def temporal_split(inter: pd.DataFrame, test_frac=0.2, val_frac=0.1) -> pd.Series:
    """Per-user time-ordered split. For each user, the latest 20% of ratings go to test,
    the 10% before those to validation, the rest to train. Returns a 'split' label per row.
    Splitting by time means we never use a later rating to predict an earlier one."""
    inter = inter.sort_values(["user_id", "ts", "recipe_id"])
    rank = inter.groupby("user_id").cumcount()
    n = inter.groupby("user_id").user_id.transform("size")
    n_test = np.maximum(1, np.floor(n * test_frac)).astype(int)
    n_val = np.maximum(1, np.floor(n * val_frac)).astype(int)
    split = np.where(rank >= n - n_test, "test", np.where(rank >= n - n_test - n_val, "val", "train"))
    return pd.Series(split, index=inter.index).reindex(inter.index)


def precision_recall_ndcg(top_k: np.ndarray, relevant: set, k: int):
    hits = np.array([1.0 if x in relevant else 0.0 for x in top_k[:k]])
    p = hits.sum() / k
    r = hits.sum() / len(relevant) if relevant else 0.0
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hits * discounts[: len(hits)]).sum()
    idcg = discounts[: min(len(relevant), k)].sum()
    return p, r, (dcg / idcg if idcg > 0 else 0.0)


def intra_list_diversity(item_idx, sim_fn) -> float:
    """1 - average pairwise cosine similarity inside one recommendation list."""
    item_idx = list(item_idx)
    if len(item_idx) < 2:
        return 0.0
    s = sim_fn(item_idx, item_idx)
    n = len(item_idx)
    return float(1 - (s.sum() - np.trace(s)) / (n * (n - 1)))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))
