"""Explain fixed-five accuracy using users held out from model fitting.

The decision-validation half was used to choose the reorder threshold, but not
to fit the evaluated model or choose boosting rounds. This script evaluates the
threshold-free top-five display on those users, without tuning on test users.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from grocery_recommender.data import load_bundle, prepare_data
from grocery_recommender.features import build_features
from grocery_recommender.metrics import rank_candidates
from grocery_recommender.modeling import split_user_ids


def _top_five_hits(candidates: pl.DataFrame, scores: np.ndarray, name: str) -> pl.DataFrame:
    return (
        rank_candidates(candidates.select("user_id", "product_id", "label"), scores)
        .filter(pl.col("rank") <= 5)
        .group_by("user_id")
        .agg(pl.col("label").sum().alias(name))
    )


def _summarize(frame: pl.DataFrame) -> dict:
    summary = {
        "users": frame.height,
        "mean_target_size": float(frame["target_size"].mean()),
        "mean_reorders": float(frame["actual_reorders"].mean()),
        "fraction_with_no_reorders": float((frame["actual_reorders"] == 0).mean()),
        "model_precision_at_5": float(frame["model_hits"].mean() / 5),
        "frequency_precision_at_5": float(frame["frequency_hits"].mean() / 5),
        "known_product_oracle_precision_at_5": float(frame["known_oracle_hits"].mean() / 5),
        "all_product_oracle_precision_at_5": float(
            frame["target_size"].clip(upper_bound=5).mean() / 5
        ),
    }
    for cutoff in ("0_5", "0_6"):
        count = int(frame[f"displayed_{cutoff}"].sum())
        summary[f"cutoff_{cutoff}"] = {
            "precision_among_displayed": (
                float(frame[f"hits_{cutoff}"].sum() / count) if count else None
            ),
            "mean_displayed_per_user": float(frame[f"displayed_{cutoff}"].mean()),
            "fraction_with_no_displayed_items": float(
                (frame[f"displayed_{cutoff}"] == 0).mean()
            ),
        }
    return summary


def run(data_dir: Path = Path("."), output_dir: Path = Path("artifacts/full_v2"),
        batch_size: int = 2000) -> dict:
    prepared = prepare_data(data_dir)
    labeled = prepared.orders.filter(pl.col("eval_set") == "train")["user_id"].to_numpy()
    validation_users = split_user_ids(labeled)[1]
    _, decision_users = train_test_split(np.sort(validation_users), test_size=0.5,
                                         random_state=42)
    model = joblib.load(output_dir / "evaluated_model.joblib")
    per_user_parts = []
    for start in range(0, len(decision_users), batch_size):
        users = decision_users[start:start + batch_size].tolist()
        bundle = load_bundle(data_dir, user_ids=users, prepared=prepared)
        features = build_features(bundle)
        candidates = features.candidates
        scores = model.predict(candidates)
        top = rank_candidates(
            candidates.select("user_id", "product_id", "label"), scores
        ).filter(pl.col("rank") <= 5)
        confidence = top.group_by("user_id").agg(
            (pl.col("score") >= 0.5).sum().alias("displayed_0_5"),
            pl.when(pl.col("score") >= 0.5).then(pl.col("label"))
            .otherwise(0).sum().alias("hits_0_5"),
            (pl.col("score") >= 0.6).sum().alias("displayed_0_6"),
            pl.when(pl.col("score") >= 0.6).then(pl.col("label"))
            .otherwise(0).sum().alias("hits_0_6"),
        )
        user = (
            candidates.group_by("user_id").agg(
                pl.col("user_order_count").first(),
                pl.col("user_unique_products").first(),
                pl.col("user_average_basket_size").first(),
                pl.len().alias("known_products"),
                pl.col("label").sum().alias("actual_reorders"),
            )
            .join(features.target_sizes, on="user_id")
            .join(_top_five_hits(candidates, scores, "model_hits"), on="user_id")
            .join(_top_five_hits(candidates,
                                 candidates["up_purchase_share"].to_numpy(),
                                 "frequency_hits"), on="user_id")
            .join(confidence, on="user_id")
            .with_columns(
                pl.min_horizontal(pl.col("actual_reorders"), pl.lit(5))
                .alias("known_oracle_hits"),
                (pl.col("target_size") - pl.col("actual_reorders"))
                .alias("new_products_in_target"),
            )
        )
        per_user_parts.append(user)
        print(f"Audited {min(start + batch_size, len(decision_users))} validation users",
              flush=True)
    per_user = pl.concat(per_user_parts)
    history_bands = (
        pl.when(pl.col("user_order_count") <= 5).then(pl.lit("02-05"))
        .when(pl.col("user_order_count") <= 10).then(pl.lit("06-10"))
        .when(pl.col("user_order_count") <= 20).then(pl.lit("11-20"))
        .otherwise(pl.lit("21+"))
    )
    basket_bands = (
        pl.when(pl.col("target_size") < 5).then(pl.lit("under 5"))
        .when(pl.col("target_size") <= 10).then(pl.lit("5-10"))
        .otherwise(pl.lit("11+"))
    )
    segments = {}
    for name, expr in (("history_orders", history_bands),
                       ("target_size", basket_bands)):
        segments[name] = {
            key: _summarize(part)
            for key, part in per_user.with_columns(expr.alias("segment"))
            .partition_by("segment", as_dict=True).items()
        }
        segments[name] = {key[0]: value for key, value in segments[name].items()}
    report = {
        "population": "decision-validation users, evaluated model",
        "overall": _summarize(per_user),
        "candidate_recall_of_target_items": float(
            per_user["actual_reorders"].sum() / per_user["target_size"].sum()
        ),
        "target_new_product_share": float(
            per_user["new_products_in_target"].sum() / per_user["target_size"].sum()
        ),
        "segments": segments,
    }
    target = output_dir / "validation_diagnostics.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
