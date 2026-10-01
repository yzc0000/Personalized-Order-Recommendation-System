"""Compare sequence features with the current model on identical validation users."""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

from grocery_recommender.data import load_bundle, prepare_data
from grocery_recommender.features import (FEATURE_COLUMNS, SEQUENCE_PILOT_COLUMNS,
                                          build_features)
from grocery_recommender.metrics import rank_candidates, ranking_metrics
from grocery_recommender.modeling import new_scorer, split_user_ids


def _load_features(data_dir: Path, prepared, users: np.ndarray,
                   batch_size: int, include_context: bool = False,
                   include_sequence_pilot: bool = False
                   ) -> tuple[pl.DataFrame, pl.DataFrame]:
    frames, sizes = [], []
    for start in range(0, len(users), batch_size):
        bundle = load_bundle(data_dir, user_ids=users[start:start + batch_size].tolist(),
                             prepared=prepared)
        result = build_features(bundle, include_context=include_context,
                                include_sequence_pilot=include_sequence_pilot)
        frames.append(result.candidates)
        sizes.append(result.target_sizes)
        print(f"Built {min(start + batch_size, len(users))}/{len(users)} users",
              flush=True)
    # Polars group-by output order varies across runs; XGBoost subsampling
    # depends on row order. Sort so small feature gains are reproducible.
    return (pl.concat(frames).sort("user_id", "product_id"),
            pl.concat(sizes).sort("user_id"))


def _hits(frame: pl.DataFrame, scores: np.ndarray) -> pl.DataFrame:
    return (
        rank_candidates(frame.select("user_id", "product_id", "label"), scores)
        .filter(pl.col("rank") <= 5).group_by("user_id")
        .agg(pl.col("label").sum().alias("hits"))
    )


def run(data_dir: Path, output: Path, max_users: int = 20000,
        batch_size: int = 2000, rounds: int = 600) -> dict:
    prepared = prepare_data(data_dir)
    eligible = prepared.orders.filter(pl.col("eval_set") == "train")["user_id"].to_numpy()
    original_training_users = split_user_ids(eligible)[0]
    sampled = np.random.default_rng(42).choice(
        original_training_users, size=min(max_users, len(original_training_users)),
        replace=False,
    )
    train_users, validation_users, _ = split_user_ids(sampled)
    train, _ = _load_features(data_dir, prepared, train_users, batch_size,
                              include_sequence_pilot=True)
    validation, sizes = _load_features(data_dir, prepared, validation_users,
                                       batch_size, include_sequence_pilot=True)
    labels = validation["label"].to_numpy()
    results = {}
    per_user = []
    for name, columns in (("current", FEATURE_COLUMNS),
                          ("sequence", FEATURE_COLUMNS + SEQUENCE_PILOT_COLUMNS)):
        scorer = new_scorer("xgboost")
        scorer.feature_columns = tuple(columns)
        scorer.estimator.set_params(n_estimators=rounds)
        print(f"Fitting {name} model on {train.height} candidates", flush=True)
        scorer.estimator.fit(train.select(columns).to_numpy().astype(np.float32),
                             train["label"].to_numpy())
        scores = scorer.predict(validation)
        results[name] = {
            **ranking_metrics(validation, sizes, scores, 5),
            "average_precision": float(average_precision_score(labels, scores)),
        }
        per_user.append(_hits(validation, scores).rename({"hits": name}))
        print(f"{name}: Precision@5={results[name]['precision_at_5']:.5f}",
              flush=True)
    paired = per_user[0].join(per_user[1], on="user_id")
    differences = (paired["sequence"] - paired["current"]).to_numpy() / 5
    rng = np.random.default_rng(42)
    bootstrap = np.array([
        rng.choice(differences, size=len(differences), replace=True).mean()
        for _ in range(2000)
    ])
    report = {
        "train_users": len(train_users),
        "validation_users": len(validation_users),
        "excluded_sample_test_users": len(sampled) - len(train_users) - len(validation_users),
        "population": "sampled only from the original full-run training users",
        "boosting_rounds": rounds,
        "results": results,
        "sequence_minus_current_precision_at_5": float(differences.mean()),
        "paired_user_bootstrap_95_percent_interval":
            [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])],
        "sequence_features": SEQUENCE_PILOT_COLUMNS,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/sequence_feature_pilot.json"))
    parser.add_argument("--max-users", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--rounds", type=int, default=600)
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.output, args.max_users,
                         args.batch_size, args.rounds), indent=2))
