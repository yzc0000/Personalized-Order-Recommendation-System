"""Evaluate time-weighted customer neighbors as a repeat-only scorer.

The neighbor index is fitted on original-training users only. Validation and
test customers are disjoint from its stored profiles. This evaluates ranking
of products customers already bought; novel retrieval is evaluated separately.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from experiments.benchmark_models import _hits, _paired_interval
from grocery_recommender.data import load_bundle, prepare_data
from grocery_recommender.features import build_features
from grocery_recommender.metrics import ranking_metrics
from grocery_recommender.modeling import new_scorer, split_user_ids


def run(data_dir: Path, retrieval_dir: Path, output_dir: Path,
        max_users: int = 10000) -> dict:
    index = joblib.load(retrieval_dir / "neighbor_index.joblib")
    prepared = prepare_data(data_dir)
    eligible = prepared.orders.filter(pl.col("eval_set") == "train")["user_id"].to_numpy()
    original_train = split_user_ids(eligible)[0]
    sampled = np.random.default_rng(2026).choice(
        original_train, size=min(max_users, len(original_train)), replace=False,
    )
    train_users, validation_users, test_users = split_user_ids(sampled)
    if not np.array_equal(index.train_users, np.sort(train_users)):
        raise ValueError("Neighbor index was fitted on a different customer split")
    bundle = load_bundle(data_dir, user_ids=sampled.tolist(), prepared=prepared)
    features = build_features(bundle)
    candidates = features.candidates.sort("user_id", "product_id")
    _, scores = index.score_users(
        bundle, np.sort(np.concatenate((validation_users, test_users))),
        candidates.select("user_id", "product_id"), per_user=1,
    )
    if scores.height != candidates.filter(
            pl.col("user_id").is_in(np.concatenate((validation_users, test_users)))) .height:
        raise AssertionError("Each held-out candidate requires one neighbor score")
    report = {
        "population": "original full-run training users, sampled with seed 2026",
        "scorer": "time-weighted personal and nearest-customer product rates",
        "settings": {"group_size": index.group_size,
                     "within_decay": index.within_decay,
                     "group_decay": index.group_decay,
                     "neighbors": index.neighbors,
                     "personal_weight": index.personal_weight},
        "train_users": len(train_users), "validation_users": len(validation_users),
        "test_users": len(test_users),
    }
    frequency = new_scorer("frequency")
    for name, users in (("validation", validation_users), ("test", test_users)):
        subset = candidates.filter(pl.col("user_id").is_in(users))
        sizes = features.target_sizes.filter(pl.col("user_id").is_in(users))
        scored = subset.join(scores, on=["user_id", "product_id"], how="left",
                             maintain_order="left")
        if scored["neighbor_score"].null_count():
            raise AssertionError("Missing held-out neighbor scores")
        neighbor_scores = scored["neighbor_score"].to_numpy()
        frequency_scores = frequency.predict(subset)
        neighbor = ranking_metrics(subset, sizes, neighbor_scores, 5)
        baseline = ranking_metrics(subset, sizes, frequency_scores, 5)
        report[name] = {
            "neighbor": neighbor, "frequency": baseline,
            "paired_p5_gain_interval": _paired_interval(
                _hits(subset, neighbor_scores, users),
                _hits(subset, frequency_scores, users),
            ),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n",
                                              encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--retrieval-dir", type=Path,
                        default=Path("artifacts/discovery_10k_ablation"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/temporal_neighbor_10k"))
    parser.add_argument("--max-users", type=int, default=10000)
    args = parser.parse_args()
    run(args.data_dir, args.retrieval_dir, args.output_dir, args.max_users)
