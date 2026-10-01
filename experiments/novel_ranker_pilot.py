"""Test whether new-to-customer products improve the same five-slot list."""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from grocery_recommender.data import load_bundle
from grocery_recommender.features import FEATURE_COLUMNS, build_features
from grocery_recommender.metrics import rank_candidates, ranking_metrics
from grocery_recommender.modeling import fit_scorer, new_scorer, split_user_ids
from grocery_recommender.novel import item_collaborative_candidates


MIXED_FEATURES = tuple(FEATURE_COLUMNS) + ("collaborative_score", "is_novel")


def _candidates(bundle, base, novel, known_scores):
    reorders = base.candidates.join(known_scores, on=["user_id", "product_id"], how="left")
    reorders = reorders.with_columns(
        pl.col("collaborative_score").fill_null(0), pl.lit(0).alias("is_novel")
    )
    user_columns = [
        "user_order_count", "user_item_count", "user_unique_products",
        "user_average_basket_size", "user_mean_days_between_orders",
    ]
    user_info = reorders.group_by("user_id").agg(
        *[pl.col(name).first().alias(name) for name in user_columns]
    )
    history = (
        bundle.history_lines.join(bundle.history_orders.select("order_id", "user_id"),
                                  on="order_id")
        .join(bundle.products.select("product_id", "department_id", "aisle_id"),
              on="product_id")
    )
    department = history.group_by("user_id", "department_id").agg(
        pl.len().alias("department_items")
    )
    aisle = history.group_by("user_id", "aisle_id").agg(
        pl.len().alias("aisle_items")
    )
    target_pairs = bundle.target_lines.join(
        bundle.target_orders.select("order_id", "user_id"), on="order_id"
    ).select("user_id", "product_id").with_columns(pl.lit(1).alias("label"))
    new_items = (
        novel.join(bundle.products.select("product_id", "product_name", "department_id",
                                          "aisle_id"), on="product_id")
        .join(bundle.product_stats, on="product_id")
        .join(user_info, on="user_id")
        .join(department, on=["user_id", "department_id"], how="left")
        .join(aisle, on=["user_id", "aisle_id"], how="left")
        .join(target_pairs, on=["user_id", "product_id"], how="left")
        .with_columns(
            (pl.col("department_items") / pl.col("user_item_count"))
            .fill_null(0).alias("user_department_share"),
            (pl.col("aisle_items") / pl.col("user_item_count"))
            .fill_null(0).alias("user_aisle_share"),
            pl.col("label").fill_null(0).cast(pl.Int8),
            pl.lit(1).alias("is_novel"),
        )
    )
    for name in FEATURE_COLUMNS:
        if name not in new_items.columns:
            new_items = new_items.with_columns(pl.lit(None).cast(pl.Float64).alias(name))
    columns = ["user_id", "product_id", "product_name", "label", *MIXED_FEATURES]
    mixed = pl.concat(
        [reorders.select(columns), new_items.select(columns)], how="vertical_relaxed"
    ).sort("user_id", "product_id")
    return reorders.sort("user_id", "product_id"), mixed


def _fit_predict(frame: pl.DataFrame, features: tuple[str, ...],
                 train_users: np.ndarray, validation_users: np.ndarray,
                 test_users: np.ndarray, sizes: pl.DataFrame):
    train = frame.filter(pl.col("user_id").is_in(train_users))
    validation = frame.filter(pl.col("user_id").is_in(validation_users))
    test = frame.filter(pl.col("user_id").is_in(test_users))
    validation_sizes = sizes.filter(pl.col("user_id").is_in(validation_users))
    test_sizes = sizes.filter(pl.col("user_id").is_in(test_users))
    model = new_scorer("xgboost")
    model.feature_columns = features
    fit_scorer(model, train)
    validation_scores = model.predict(validation)
    validation_metrics = ranking_metrics(validation, validation_sizes, validation_scores, 5)
    final = new_scorer("xgboost")
    final.feature_columns = features
    fit_scorer(final, pl.concat([train, validation]))
    test_scores = final.predict(test)
    test_metrics = ranking_metrics(test, test_sizes, test_scores, 5)
    ranked = rank_candidates(test.select("user_id", "product_id", "label", "is_novel"),
                             test_scores).filter(pl.col("rank") <= 5)
    novel_top = ranked.filter(pl.col("is_novel") == 1)
    novel_mask = test["is_novel"].to_numpy() == 1
    novel_only = (
        ranking_metrics(test.filter(pl.col("is_novel") == 1), test_sizes,
                        test_scores[novel_mask], 5)
        if novel_mask.any() else None
    )
    return {
        "validation": validation_metrics,
        "test": test_metrics,
        "test_top_five_novel_items": novel_top.height,
        "test_top_five_novel_hits": int(novel_top["label"].sum()),
        "test_novel_only_five": novel_only,
    }


def run(data_dir: Path, max_users: int, top_products: int, novel_per_user: int,
        output: Path) -> dict:
    bundle = load_bundle(data_dir, max_users=max_users)
    base = build_features(bundle)
    users = np.sort(base.target_sizes["user_id"].to_numpy())
    print("Retrieving new-to-customer candidates...", flush=True)
    novel, known_scores = item_collaborative_candidates(
        base.candidates.select("user_id", "product_id"), bundle.product_stats,
        users, top_products=top_products, per_user=novel_per_user,
    )
    reorders, mixed = _candidates(bundle, base, novel, known_scores)
    train_users, validation_users, test_users = split_user_ids(users)
    print("Fitting reorder-only comparison...", flush=True)
    reorder_result = _fit_predict(
        reorders, tuple(FEATURE_COLUMNS), train_users, validation_users, test_users,
        base.target_sizes,
    )
    print("Fitting reorder-only model with collaborative score...", flush=True)
    reorder_with_collab = _fit_predict(
        reorders, MIXED_FEATURES, train_users, validation_users, test_users,
        base.target_sizes,
    )
    print("Fitting combined repeat-and-new model...", flush=True)
    mixed_result = _fit_predict(
        mixed, MIXED_FEATURES, train_users, validation_users, test_users,
        base.target_sizes,
    )
    novel_labeled = mixed.filter(pl.col("is_novel") == 1)
    novel_hits = int(novel_labeled["label"].sum())
    validation_gain = (mixed_result["validation"]["precision_at_5"] -
                       reorder_result["validation"]["precision_at_5"])
    result = {
        "users": len(users),
        "split_users": {"train": len(train_users), "validation": len(validation_users),
                        "test": len(test_users)},
        "top_products": top_products,
        "novel_per_user": novel_per_user,
        "reorder_candidate_recall": base.candidate_hits / base.target_item_count,
        "mixed_candidate_recall": (base.candidate_hits + novel_hits) / base.target_item_count,
        "novel_candidates": novel_labeled.height,
        "novel_candidate_hit_rate": novel_hits / novel_labeled.height,
        "reorder_only": reorder_result,
        "reorder_with_collaborative_score": reorder_with_collab,
        "mixed": mixed_result,
        "mixed_validation_precision_at_5_gain": validation_gain,
        "minimum_material_validation_gain": 0.005,
        "promote_mixed_to_main_top_five": validation_gain >= 0.005,
        "selected_by_validation_precision_at_5": max(
            ("reorder_only", "reorder_with_collaborative_score", "mixed"),
            key=lambda name: {
                "reorder_only": reorder_result,
                "reorder_with_collaborative_score": reorder_with_collab,
                "mixed": mixed_result,
            }[name]["validation"]["precision_at_5"],
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--max-users", type=int, default=5000)
    parser.add_argument("--top-products", type=int, default=5000)
    parser.add_argument("--novel-per-user", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("artifacts/novel_ranker_pilot.json"))
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.max_users, args.top_products,
                         args.novel_per_user, args.output), indent=2))
