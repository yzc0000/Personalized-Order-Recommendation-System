"""Measure order-start timing value without using next-basket contents.

The context columns are available only when a customer starts the next order;
this is a distinct serving scenario from pre-order recommendations.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from grocery_recommender.data import prepare_data
from grocery_recommender.features import CONTEXT_FEATURE_COLUMNS, FEATURE_COLUMNS
from grocery_recommender.metrics import ranking_metrics
from grocery_recommender.modeling import new_scorer, split_user_ids

from .sequence_feature_pilot import _hits, _load_features


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
                              include_context=True)
    validation, sizes = _load_features(data_dir, prepared, validation_users,
                                       batch_size, include_context=True)
    results = {}
    per_user = []
    for name, columns in (("pre_order", FEATURE_COLUMNS),
                          ("order_start", CONTEXT_FEATURE_COLUMNS)):
        scorer = new_scorer("xgboost")
        scorer.feature_columns = tuple(columns)
        scorer.estimator.set_params(n_estimators=rounds)
        print(f"Fitting {name} model on {train.height} candidates", flush=True)
        scorer.estimator.fit(train.select(columns).to_numpy().astype(np.float32),
                             train["label"].to_numpy())
        scores = scorer.predict(validation)
        results[name] = ranking_metrics(validation, sizes, scores, 5)
        per_user.append(_hits(validation, scores).rename({"hits": name}))
        print(f"{name}: Precision@5={results[name]['precision_at_5']:.5f}",
              flush=True)
    paired = per_user[0].join(per_user[1], on="user_id")
    differences = (paired["order_start"] - paired["pre_order"]).to_numpy() / 5
    rng = np.random.default_rng(42)
    bootstrap = np.array([
        rng.choice(differences, size=len(differences), replace=True).mean()
        for _ in range(2000)
    ])
    report = {
        "population": "sampled only from the original full-run training users",
        "train_users": len(train_users),
        "validation_users": len(validation_users),
        "boosting_rounds": rounds,
        "results": results,
        "order_start_minus_pre_order_precision_at_5": float(differences.mean()),
        "paired_user_bootstrap_95_percent_interval":
            [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])],
        "order_start_inputs": [column for column in CONTEXT_FEATURE_COLUMNS
                               if column not in FEATURE_COLUMNS],
        "serving_requirement": "next order has started; day, hour, and elapsed days are known",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/context_feature_pilot.json"))
    parser.add_argument("--max-users", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--rounds", type=int, default=600)
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.output, args.max_users,
                         args.batch_size, args.rounds), indent=2))
