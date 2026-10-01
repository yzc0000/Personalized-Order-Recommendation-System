"""Evaluate confidence cutoffs for the up-to-five reorder display.

Thresholds are evaluated only after ranking each customer's repeat candidates
and keeping the top five. The decision-validation half is used to compare
operating points; the established test split is reported for context.
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


THRESHOLDS = (0.5, 0.6, 0.75, 0.9)


def _summarize(users: int, eligible: int, actual_repeat_items: int,
               shown: int, hits: int) -> dict:
    precision = hits / shown if shown else 0.0
    recall = hits / actual_repeat_items if actual_repeat_items else 0.0
    return {
        "customers": users,
        "customers_with_at_least_one_suggestion": eligible,
        "customers_with_no_suggestions": users - eligible,
        "share_with_no_suggestions": (users - eligible) / users,
        "suggestions_shown": shown,
        "mean_suggestions_per_customer": shown / users,
        "matching_repeat_items": hits,
        "precision_among_suggestions": precision,
        "repeat_item_recall_micro": recall,
        "f1_micro": (2 * precision * recall / (precision + recall)
                     if precision + recall else 0.0),
    }


def _evaluate_population(data_dir: Path, prepared, users: np.ndarray,
                         model, batch_size: int) -> dict:
    accum = {
        threshold: {"eligible": 0, "shown": 0, "hits": 0,
                    "actual_repeat_items": 0}
        for threshold in THRESHOLDS
    }
    for start in range(0, len(users), batch_size):
        batch_users = users[start:start + batch_size]
        bundle = load_bundle(data_dir, user_ids=batch_users.tolist(), prepared=prepared)
        candidates = build_features(bundle).candidates
        scores = model.predict(candidates)
        ranked = rank_candidates(
            candidates.select("user_id", "product_id", "label"), scores,
        ).filter(pl.col("rank") <= 5)
        actual_repeat_items = int(candidates["label"].sum())
        for threshold in THRESHOLDS:
            selected = ranked.filter(pl.col("score") >= threshold)
            by_user = selected.group_by("user_id").agg(
                pl.len().alias("shown"), pl.col("label").sum().alias("hits"),
            )
            accum[threshold]["eligible"] += by_user.height
            accum[threshold]["shown"] += int(selected.height)
            accum[threshold]["hits"] += int(selected["label"].sum()) if selected.height else 0
            accum[threshold]["actual_repeat_items"] += actual_repeat_items
        print(f"Scored {min(start + batch_size, len(users))}/{len(users)} customers",
              flush=True)

    output = {}
    for threshold, values in accum.items():
        output[f"{threshold:.2f}"] = _summarize(
            users=len(users), eligible=values["eligible"],
            actual_repeat_items=values["actual_repeat_items"],
            shown=values["shown"], hits=values["hits"],
        )
    return output


def run(data_dir: Path = Path("."), model_dir: Path = Path("artifacts/full_v2"),
        output: Path = Path("artifacts/full_v2/threshold_display_metrics.json"),
        batch_size: int = 2000) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    prepared = prepare_data(data_dir)
    labeled_users = prepared.orders.filter(pl.col("eval_set") == "train")[
        "user_id"
    ].to_numpy()
    _, validation_users, test_users = split_user_ids(labeled_users)
    _, decision_users = train_test_split(
        np.sort(validation_users), test_size=0.5, random_state=42,
    )
    model = joblib.load(model_dir / "evaluated_model.joblib")
    result = {
        "model": "evaluated_model.joblib",
        "display_policy": "rank repeat candidates, keep top five, then apply score cutoff",
        "thresholds": list(THRESHOLDS),
        "decision_validation": _evaluate_population(
            data_dir, prepared, decision_users, model, batch_size,
        ),
        "established_test": _evaluate_population(
            data_dir, prepared, test_users, model, batch_size,
        ),
        "test_status": (
            "Historical benchmark: this test split has been inspected in prior "
            "project analysis and is not an untouched confirmation set."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/full_v2"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/full_v2/threshold_display_metrics.json"))
    parser.add_argument("--batch-size", type=int, default=2000)
    args = parser.parse_args()
    run(args.data_dir, args.model_dir, args.output, args.batch_size)
