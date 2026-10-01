"""Classification and next-basket ranking metrics."""

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


def classification_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    if np.unique(labels).size < 2:
        raise ValueError("Both label classes are needed to evaluate a model")
    predicted = scores >= 0.5
    return {
        "pr_auc": float(average_precision_score(labels, scores)),
        "precision_at_0_5": float(precision_score(labels, predicted, zero_division=0)),
        "recall_at_0_5": float(recall_score(labels, predicted, zero_division=0)),
        "f1_at_0_5": float(f1_score(labels, predicted, zero_division=0)),
    }


def rank_candidates(candidates: pl.DataFrame, scores: np.ndarray) -> pl.DataFrame:
    if len(scores) != candidates.height:
        raise ValueError("Each candidate needs one score")
    return (
        candidates.with_columns(pl.Series("score", scores))
        .sort(["user_id", "score", "product_id"], descending=[False, True, False])
        .with_columns(pl.col("user_id").cum_count().over("user_id").alias("rank"))
    )


def ranking_metrics(
    candidates: pl.DataFrame, target_sizes: pl.DataFrame, scores: np.ndarray, k: int
) -> dict[str, float]:
    if k < 1:
        raise ValueError("k must be positive")
    ranked = rank_candidates(candidates.select("user_id", "product_id", "label"), scores)
    per_user = (
        ranked.filter(pl.col("rank") <= k)
        .with_columns(
            (pl.col("label") / (pl.col("rank") + 1).cast(pl.Float64).log(base=2))
            .alias("discounted_hit")
        )
        .group_by("user_id")
        .agg(pl.col("label").sum().alias("hits"), pl.col("discounted_hit").sum().alias("dcg"))
    )
    totals = target_sizes.join(per_user, on="user_id", how="left").with_columns(
        pl.col("hits").fill_null(0), pl.col("dcg").fill_null(0)
    )
    target = totals["target_size"].to_numpy()
    hits = totals["hits"].to_numpy()
    dcg = totals["dcg"].to_numpy()
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    ideal = np.concatenate(([0.0], np.cumsum(discounts)))
    return {
        f"precision_at_{k}": float(np.mean(hits / k)),
        f"recall_at_{k}": float(np.mean(hits / target)),
        f"ndcg_at_{k}": float(np.mean(dcg / ideal[np.minimum(target, k)])),
    }
