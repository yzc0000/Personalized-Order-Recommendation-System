"""Test whether temporal customer-neighbor scores improve XGBoost reorders."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from experiments.benchmark_models import _hits, _paired_interval
from grocery_recommender.data import load_bundle, prepare_data
from grocery_recommender.features import FEATURE_COLUMNS, build_features
from grocery_recommender.metrics import ranking_metrics
from grocery_recommender.modeling import fit_scorer, new_scorer, split_user_ids


def run(data_dir: Path, retrieval_dir: Path, output_dir: Path,
        max_users: int = 10000, rounds: int = 350) -> dict:
    index = joblib.load(retrieval_dir / "neighbor_index.joblib")
    prepared = prepare_data(data_dir)
    eligible = prepared.orders.filter(pl.col("eval_set") == "train")["user_id"].to_numpy()
    original_train = split_user_ids(eligible)[0]
    sampled = np.random.default_rng(2026).choice(
        original_train, size=min(max_users, len(original_train)), replace=False,
    )
    train_users, validation_users, test_users = split_user_ids(sampled)
    if not np.array_equal(index.train_users, np.sort(train_users)):
        raise ValueError("Neighbor index and model must use the same training users")
    bundle = load_bundle(data_dir, user_ids=sampled.tolist(), prepared=prepared)
    features = build_features(bundle)
    base = features.candidates.sort("user_id", "product_id")
    print(f"Computing leave-one-user-out neighbor scores for {len(sampled)} users",
          flush=True)
    _, known_scores = index.score_users(
        bundle, np.sort(sampled), base.select("user_id", "product_id"), per_user=1,
    )
    rich = base.join(known_scores, on=["user_id", "product_id"], how="left",
                     maintain_order="left")
    if rich.height != base.height or rich["neighbor_score"].null_count():
        raise AssertionError("Neighbor scores must cover every reorder candidate")

    def fit(frame: pl.DataFrame, users: np.ndarray, extra: bool):
        model = new_scorer("xgboost")
        model.feature_columns = tuple(FEATURE_COLUMNS) + (("neighbor_score",) if extra else ())
        model.estimator.set_params(n_estimators=rounds)
        return fit_scorer(model, frame.filter(pl.col("user_id").is_in(users)))

    print("Fitting baseline and neighbor-feature classifiers", flush=True)
    baseline = fit(base, train_users, False)
    challenger = fit(rich, train_users, True)
    val_base = base.filter(pl.col("user_id").is_in(validation_users))
    val_rich = rich.filter(pl.col("user_id").is_in(validation_users))
    val_sizes = features.target_sizes.filter(pl.col("user_id").is_in(validation_users))
    base_scores = baseline.predict(val_base)
    new_scores = challenger.predict(val_rich)
    baseline_p5 = ranking_metrics(val_base, val_sizes, base_scores, 5)["precision_at_5"]
    neighbor_p5 = ranking_metrics(val_rich, val_sizes, new_scores, 5)["precision_at_5"]
    interval = _paired_interval(_hits(val_rich, new_scores, validation_users),
                                _hits(val_base, base_scores, validation_users))
    selected = ("neighbor_feature" if neighbor_p5 - baseline_p5 >= .005 and
                interval[0] > 0 else "baseline")
    fit_users = np.concatenate((train_users, validation_users))
    final = fit(rich if selected == "neighbor_feature" else base, fit_users,
                selected == "neighbor_feature")
    test = (rich if selected == "neighbor_feature" else base).filter(
        pl.col("user_id").is_in(test_users))
    test_sizes = features.target_sizes.filter(pl.col("user_id").is_in(test_users))
    test_metrics = ranking_metrics(test, test_sizes, final.predict(test), 5)
    result = {
        "population": "same 10k original-training-user sample as model benchmark",
        "feature_sources": "completed orders only; own training user's neighbor profile excluded",
        "train_users": len(train_users), "validation_users": len(validation_users),
        "test_users": len(test_users), "rounds": rounds,
        "validation": {"baseline_p5": baseline_p5,
                       "neighbor_feature_p5": neighbor_p5,
                       "gain": neighbor_p5 - baseline_p5,
                       "paired_95pct_interval": interval},
        "promotion_rule": "+0.005 absolute validation P@5 and positive paired interval",
        "selected": selected, "selected_test": test_metrics,
        "note": "This pilot does not automatically replace the full-run model.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n",
                                              encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--retrieval-dir", type=Path,
                        default=Path("artifacts/discovery_10k_ablation"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/collaborative_feature_10k"))
    parser.add_argument("--max-users", type=int, default=10000)
    parser.add_argument("--rounds", type=int, default=350)
    args = parser.parse_args()
    run(args.data_dir, args.retrieval_dir, args.output_dir, args.max_users,
        args.rounds)
