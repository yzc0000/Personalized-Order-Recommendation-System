"""Compare current-order training with leakage-controlled historical prefixes."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from experiments.benchmark_models import _hits, _paired_interval
from grocery_recommender.augmentation import historical_training_examples
from grocery_recommender.data import load_bundle, prepare_data
from grocery_recommender.features import FEATURE_COLUMNS, build_features
from grocery_recommender.metrics import ranking_metrics
from grocery_recommender.modeling import fit_scorer, new_scorer, split_user_ids


# Full-catalog product popularity is computed using all prior orders, including
# dates after a pseudo-target. Exclude both global features for this experiment.
HISTORY_SAFE_COLUMNS = tuple(c for c in FEATURE_COLUMNS if c not in (
    "product_purchase_count", "product_reorder_rate",
))


def _fit(frame: pl.DataFrame, rounds: int):
    scorer = new_scorer("xgboost")
    scorer.feature_columns = HISTORY_SAFE_COLUMNS
    scorer.estimator.set_params(n_estimators=rounds)
    return fit_scorer(scorer, frame)


def run(data_dir: Path, output_dir: Path, max_users: int = 5000,
        snapshots: int = 3, rounds: int = 350) -> dict:
    if max_users < 100 or snapshots < 1 or rounds < 1:
        raise ValueError("Use at least 100 users and positive snapshots and rounds")
    prepared = prepare_data(data_dir)
    eligible = prepared.orders.filter(pl.col("eval_set") == "train")["user_id"].to_numpy()
    original_train = split_user_ids(eligible)[0]
    sampled = np.random.default_rng(2026).choice(
        original_train, size=min(max_users, len(original_train)), replace=False,
    )
    train_users, validation_users, test_users = split_user_ids(sampled)
    print(f"Loading {len(sampled)} users for historical-prefix ablation", flush=True)
    bundle = load_bundle(data_dir, user_ids=sampled.tolist(), prepared=prepared)
    features = build_features(bundle)
    base = features.candidates.sort("user_id", "product_id")
    train = base.filter(pl.col("user_id").is_in(train_users))
    validation = base.filter(pl.col("user_id").is_in(validation_users))
    val_sizes = features.target_sizes.filter(pl.col("user_id").is_in(validation_users))
    pseudo = historical_training_examples(bundle, train_users, snapshots)
    augmented = pl.concat([
        train.with_columns(pl.lit(0, dtype=pseudo.schema["snapshot_id"]).alias("snapshot_id")),
        pseudo.select(train.columns + ["snapshot_id"]),
    ]).sort("user_id", "snapshot_id", "product_id")
    print(f"Fitting baseline on {train.height} candidate rows", flush=True)
    baseline = _fit(train, rounds)
    print(f"Fitting augmented model on {augmented.height} candidate rows", flush=True)
    challenger = _fit(augmented, rounds)
    base_scores = baseline.predict(validation)
    new_scores = challenger.predict(validation)
    baseline_p5 = ranking_metrics(validation, val_sizes, base_scores, 5)["precision_at_5"]
    augmented_p5 = ranking_metrics(validation, val_sizes, new_scores, 5)["precision_at_5"]
    interval = _paired_interval(_hits(validation, new_scores, validation_users),
                                _hits(validation, base_scores, validation_users))
    selected = ("augmented" if augmented_p5 - baseline_p5 >= .005 and interval[0] > 0
                else "baseline")
    print(f"Validation baseline={baseline_p5:.5f}, prefixes={augmented_p5:.5f}; "
          f"selected={selected}", flush=True)
    final_users = np.concatenate((train_users, validation_users))
    fit_base = base.filter(pl.col("user_id").is_in(final_users))
    if selected == "augmented":
        more_pseudo = historical_training_examples(bundle, validation_users, snapshots)
        fit_frame = pl.concat([
            fit_base.with_columns(pl.lit(0, dtype=pseudo.schema["snapshot_id"])
                                  .alias("snapshot_id")),
            pseudo.select(fit_base.columns + ["snapshot_id"]),
            more_pseudo.select(fit_base.columns + ["snapshot_id"]),
        ]).sort("user_id", "snapshot_id", "product_id")
    else:
        fit_frame = fit_base
    final = _fit(fit_frame, rounds)
    test = base.filter(pl.col("user_id").is_in(test_users))
    test_sizes = features.target_sizes.filter(pl.col("user_id").is_in(test_users))
    test_scores = final.predict(test)
    report = {
        "population": "sample of original full-run training users",
        "decision_time": "before next order starts",
        "training_feature_columns": list(HISTORY_SAFE_COLUMNS),
        "global_product_features_excluded": ["product_purchase_count",
                                             "product_reorder_rate"],
        "historical_snapshots_per_fit_user_requested": snapshots,
        "split_users": {"train": len(train_users), "validation": len(validation_users),
                        "test": len(test_users)},
        "train_rows": {"current_only": train.height, "with_prefixes": augmented.height,
                       "pseudo": pseudo.height},
        "validation": {"current_only_p5": baseline_p5,
                       "with_prefixes_p5": augmented_p5,
                       "gain": augmented_p5 - baseline_p5,
                       "paired_95pct_interval": interval},
        "promotion_rule": "+0.005 absolute validation P@5 and positive paired interval",
        "selected": selected,
        "selected_test": ranking_metrics(test, test_sizes, test_scores, 5),
        "note": "This pilot does not automatically replace the full-run model.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n",
                                              encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/historical_prefix_5k"))
    parser.add_argument("--max-users", type=int, default=5000)
    parser.add_argument("--snapshots", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=350)
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.max_users, args.snapshots,
        args.rounds)
