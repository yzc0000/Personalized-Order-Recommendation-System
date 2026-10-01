"""Save a reproducible Top-K example for the grocery project report."""

import argparse
import json
from pathlib import Path

import joblib
import polars as pl

from grocery_recommender.data import load_bundle, validate_bundle
from grocery_recommender.features import REPLENISHMENT_PILOT_COLUMNS, build_features
from grocery_recommender.metrics import rank_candidates


def run(data_dir: Path, model_dir: Path, output: Path,
        user_id: int = 3, k: int = 5,
        min_probability: float = 0.5) -> dict:
    if k < 1:
        raise ValueError("k must be positive")
    if not 0 <= min_probability <= 1:
        raise ValueError("min_probability must be between 0 and 1")
    model = joblib.load(model_dir / "model.joblib")
    if model.name in {"frequency", "xgboost_ranker"}:
        raise ValueError("The portfolio example requires a probability model")
    bundle = load_bundle(data_dir, user_id=user_id)
    validate_bundle(bundle)
    needs_replenishment = any(
        name in model.feature_columns for name in REPLENISHMENT_PILOT_COLUMNS
    )
    candidates = build_features(
        bundle, include_replenishment_pilot=needs_replenishment,
    ).candidates
    ranked = rank_candidates(
        candidates.select("user_id", "product_id", "product_name"),
        model.predict(candidates),
    ).filter((pl.col("rank") <= k) &
             (pl.col("score") >= min_probability))
    example = {
        "user_id": user_id,
        "prediction_time": "before next order starts",
        "model": "all-labeled deployment fit; held-out metrics belong to evaluated_model.joblib",
        "score_meaning": "estimated probability of reordering this product in the next basket",
        "minimum_probability": min_probability,
        "completed_orders": bundle.history_orders.height,
        "previously_purchased_product_candidates": candidates.height,
        "display_limit": k,
        "recommendations": [
            {"rank": int(row["rank"]), "product_id": int(row["product_id"]),
             "product_name": row["product_name"],
             "estimated_probability": float(row["score"])}
            for row in ranked.to_dicts()
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(example, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(example, indent=2), flush=True)
    return example


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/full_v2"))
    parser.add_argument("--output", type=Path,
                        default=Path("reports/example_recommendations.json"))
    parser.add_argument("--user-id", type=int, default=3)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min-probability", type=float, default=0.5)
    args = parser.parse_args()
    run(args.data_dir, args.model_dir, args.output, args.user_id, args.k,
        args.min_probability)
