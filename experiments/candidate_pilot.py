"""Measure whether simple unseen-product candidates can improve basket coverage.

This is an exploratory pilot. It never uses next-basket items to generate candidates.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from grocery_recommender.data import load_bundle
from grocery_recommender.features import build_features


def category_candidates(
    history: pl.DataFrame, known: pl.DataFrame, targets: pl.DataFrame,
    product_stats: pl.DataFrame, products: pl.DataFrame, category: str,
) -> dict[str, float | int]:
    preferred = (
        history.group_by("user_id", category).agg(pl.len().alias("count"))
        .sort(["user_id", "count", category], descending=[False, True, False])
        .with_columns(pl.col("user_id").cum_count().over("user_id").alias("rank"))
        .filter(pl.col("rank") <= 3).select("user_id", category)
    )
    popular = (
        product_stats.join(products.select("product_id", category), on="product_id")
        .sort([category, "product_purchase_count", "product_id"],
              descending=[False, True, False])
        .with_columns(pl.col(category).cum_count().over(category).alias("rank"))
        .filter(pl.col("rank") <= 10).select(category, "product_id")
    )
    extra = (
        preferred.join(popular, on=category).select("user_id", "product_id").unique()
        .join(known, on=["user_id", "product_id"], how="anti")
    )
    hits = extra.join(targets, on=["user_id", "product_id"]).height
    return {"extra_candidates": extra.height, "extra_hits": hits,
            "coverage_gain": hits / targets.height, "novel_precision": hits / extra.height}


def collaborative_candidates(
    known: pl.DataFrame, targets: pl.DataFrame,
    product_stats: pl.DataFrame, users: np.ndarray, top_products: int,
    novel_per_user: int,
) -> dict[str, float | int]:
    product_ids = product_stats.sort("product_purchase_count", descending=True).head(top_products)[
        "product_id"
    ].to_numpy()
    user_index = {int(value): index for index, value in enumerate(users)}
    product_index = {int(value): index for index, value in enumerate(product_ids)}
    observed = known.filter(pl.col("product_id").is_in(product_ids))
    rows = np.fromiter((user_index[int(value)] for value in observed["user_id"]),
                       dtype=np.int32, count=observed.height)
    columns = np.fromiter((product_index[int(value)] for value in observed["product_id"]),
                          dtype=np.int32, count=observed.height)
    incidence = csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(len(users), len(product_ids)),
    )
    cooccurrence = (incidence.T @ incidence).toarray()
    frequency = np.diag(cooccurrence).copy()
    np.fill_diagonal(cooccurrence, 0)
    cooccurrence /= np.sqrt(np.maximum(frequency[:, None] * frequency[None, :], 1))
    scores = incidence @ cooccurrence
    scores[rows, columns] = -np.inf
    top = np.argpartition(scores, -novel_per_user, axis=1)[:, -novel_per_user:]
    user_rows = np.repeat(np.arange(len(users)), novel_per_user)
    product_columns = top.reshape(-1)
    keep = np.isfinite(scores[user_rows, product_columns]) & (
        scores[user_rows, product_columns] > 0
    )
    extra = (
        pl.DataFrame({
            "user_id": users[user_rows[keep]],
            "product_id": product_ids[product_columns[keep]],
        }).unique().join(known, on=["user_id", "product_id"], how="anti")
    )
    hits = extra.join(targets, on=["user_id", "product_id"]).height
    return {"extra_candidates": extra.height, "extra_hits": hits,
            "coverage_gain": hits / targets.height, "novel_precision": hits / extra.height}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot unseen-product candidate methods")
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--max-users", type=int, default=5000)
    parser.add_argument("--top-products", type=int, default=5000)
    parser.add_argument("--novel-per-user", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("artifacts/candidate_pilot.json"))
    args = parser.parse_args()
    if not 1 <= args.novel_per_user < args.top_products <= 5000:
        raise ValueError("Require 1 <= novel_per_user < top_products <= 5000")
    bundle = load_bundle(args.data_dir, max_users=args.max_users)
    features = build_features(bundle)
    history = (
        bundle.history_lines.join(bundle.history_orders.select("order_id", "user_id"), on="order_id")
        .join(bundle.products.select("product_id", "department_id", "aisle_id"), on="product_id")
    )
    known = features.candidates.select("user_id", "product_id")
    targets = (
        bundle.target_lines.join(bundle.target_orders.select("order_id", "user_id"), on="order_id")
        .select("user_id", "product_id")
    )
    users = np.sort(bundle.target_orders["user_id"].to_numpy())
    result = {
        "sample_users": len(users), "next_basket_items": targets.height,
        "existing_candidate_recall": features.candidate_hits / targets.height,
        "department_popularity": category_candidates(
            history, known, targets, bundle.product_stats, bundle.products, "department_id"
        ),
        "aisle_popularity": category_candidates(
            history, known, targets, bundle.product_stats, bundle.products, "aisle_id"
        ),
        "item_collaborative": collaborative_candidates(
            known, targets, bundle.product_stats, users,
            args.top_products, args.novel_per_user,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
