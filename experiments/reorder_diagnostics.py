"""Audit the held-out reorder model's scores and fixed-five display."""

import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             log_loss, precision_score, recall_score)

from grocery_recommender.data import load_bundle, prepare_data
from grocery_recommender.features import build_features
from grocery_recommender.modeling import split_user_ids


def _bins(labels: np.ndarray, scores: np.ndarray) -> list[dict]:
    indices = np.minimum((scores * 10).astype(np.int32), 9)
    return [
        {
            "lower": index / 10,
            "upper": (index + 1) / 10,
            "count": int(mask.sum()),
            "mean_score": float(scores[mask].mean()) if mask.any() else None,
            "hit_rate": float(labels[mask].mean()) if mask.any() else None,
        }
        for index in range(10)
        for mask in [indices == index]
    ]


def run(data_dir: Path = Path("."), output_dir: Path = Path("artifacts/full"),
        batch_size: int = 5000) -> dict:
    prepared = prepare_data(data_dir)
    labeled = prepared.orders.filter(pl.col("eval_set") == "train")["user_id"].to_numpy()
    test_users = split_user_ids(labeled)[2]
    model = joblib.load(output_dir / "evaluated_model.joblib")
    labels_parts = []
    score_parts = []
    per_user_parts = []
    for start in range(0, len(test_users), batch_size):
        users = test_users[start:start + batch_size].tolist()
        bundle = load_bundle(data_dir, user_ids=users, prepared=prepared)
        features = build_features(bundle)
        candidates = features.candidates
        scores = model.predict(candidates)
        labels_parts.append(candidates["label"].to_numpy())
        score_parts.append(scores)
        per_user_parts.append(
            candidates.select("user_id", "label").with_columns(
                pl.Series("predicted_at_0_5", (scores >= 0.5).astype(np.int8))
            ).group_by("user_id").agg(
                pl.len().alias("known_products"),
                pl.col("label").sum().alias("actual_reorders"),
                pl.col("predicted_at_0_5").sum().alias("predicted_reorders_at_0_5"),
            ).join(features.target_sizes, on="user_id")
        )
        print(f"Audited {min(start + batch_size, len(test_users))} test users", flush=True)
    labels = np.concatenate(labels_parts)
    scores = np.concatenate(score_parts)
    per_user = pl.concat(per_user_parts)
    predicted = scores >= 0.5
    top_five = pl.read_csv(output_dir / "test_recommendations.csv")
    top_labels = top_five["label"].to_numpy()
    top_scores = top_five["score"].to_numpy()
    report = {
        "test_users": len(test_users),
        "candidate_pairs": len(labels),
        "candidate_positive_rate": float(labels.mean()),
        "item_level": {
            "average_precision": float(average_precision_score(labels, scores)),
            "brier": float(brier_score_loss(labels, scores)),
            "constant_prevalence_brier": float(labels.mean() * (1 - labels.mean())),
            "log_loss": float(log_loss(labels, scores)),
            "calibration_bins": _bins(labels, scores),
        },
        "untuned_threshold_0_5": {
            "precision": float(precision_score(labels, predicted, zero_division=0)),
            "recall": float(recall_score(labels, predicted, zero_division=0)),
            "f1": float(f1_score(labels, predicted, zero_division=0)),
            "mean_predicted_reorders": float(per_user["predicted_reorders_at_0_5"].mean()),
            "users_with_zero_predictions": int(per_user.filter(
                pl.col("predicted_reorders_at_0_5") == 0).height),
        },
        "basket_size": {
            "mean": float(per_user["target_size"].mean()),
            "users_with_fewer_than_five_items": int(per_user.filter(
                pl.col("target_size") < 5).height),
            "users_with_fewer_than_five_known_products": int(per_user.filter(
                pl.col("known_products") < 5).height),
        },
        "displayed_top_five": {
            "recommendations": len(top_labels),
            "mean_score": float(top_scores.mean()),
            "hit_rate_among_displayed": float(top_labels.mean()),
            "calibration_bins": _bins(top_labels, top_scores),
            "note": "Display hit rate divides by displayed items; Precision@5 divides by five per user.",
        },
    }
    target = output_dir / "reorder_diagnostics.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"item_level", "displayed_top_five"}}, indent=2))
    print("Saved full diagnostics to artifacts/full/reorder_diagnostics.json")
