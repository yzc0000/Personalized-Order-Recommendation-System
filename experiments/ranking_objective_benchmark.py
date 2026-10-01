"""Bounded test of equal-user weighting and a top-five ranking objective."""

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


def _fit(frame: pl.DataFrame, kind: str, rounds: int):
    model = new_scorer("xgboost_ranker" if kind == "top5_ranker" else "xgboost")
    model.feature_columns = tuple(FEATURE_COLUMNS)
    model.estimator.set_params(n_estimators=rounds)
    if kind == "top5_ranker":
        model.estimator.set_params(
            eval_metric="ndcg@5", lambdarank_pair_method="topk",
            lambdarank_num_pair_per_sample=5,
        )
    if kind == "user_weighted":
        counts = frame.group_by("user_id").len().rename({"len": "candidate_count"})
        weighted = frame.join(counts, on="user_id", how="left",
                              maintain_order="left")
        mean_count = frame.height / counts.height
        weights = (mean_count / weighted["candidate_count"].to_numpy()).astype(np.float32)
        model.estimator.fit(model.matrix(weighted), weighted["label"].to_numpy(),
                            sample_weight=weights)
        return model
    return fit_scorer(model, frame)


def run(data_dir: Path, output_dir: Path, max_users: int = 10000,
        rounds: int = 350) -> dict:
    if max_users < 100 or rounds < 1:
        raise ValueError("Use at least 100 users and positive rounds")
    prepared = prepare_data(data_dir)
    eligible = prepared.orders.filter(pl.col("eval_set") == "train")["user_id"].to_numpy()
    original_train = split_user_ids(eligible)[0]
    sampled = np.random.default_rng(2026).choice(
        original_train, size=min(max_users, len(original_train)), replace=False,
    )
    train_users, validation_users, test_users = split_user_ids(sampled)
    bundle = load_bundle(data_dir, user_ids=sampled.tolist(), prepared=prepared)
    features = build_features(bundle)
    candidates = features.candidates.sort("user_id", "product_id")
    train = candidates.filter(pl.col("user_id").is_in(train_users))
    validation = candidates.filter(pl.col("user_id").is_in(validation_users))
    val_sizes = features.target_sizes.filter(pl.col("user_id").is_in(validation_users))
    variants = ("classifier", "user_weighted", "top5_ranker")
    val_result = {}
    baseline_hits = None
    baseline_p5 = None
    for kind in variants:
        print(f"Fitting {kind} on {len(train_users)} users", flush=True)
        model = _fit(train, kind, rounds)
        scores = model.predict(validation)
        p5 = ranking_metrics(validation, val_sizes, scores, 5)["precision_at_5"]
        hits = _hits(validation, scores, validation_users)
        if kind == "classifier":
            baseline_hits, baseline_p5 = hits, p5
        val_result[kind] = {
            "precision_at_5": p5,
            "gain_vs_classifier": p5 - baseline_p5,
            "paired_95pct_interval": _paired_interval(hits, baseline_hits),
        }
        print(f"{kind} validation P@5={p5:.5f}", flush=True)
    shortlisted = [kind for kind in variants[1:] if
                   val_result[kind]["gain_vs_classifier"] >= .005 and
                   val_result[kind]["paired_95pct_interval"][0] > 0]
    selected = (max(shortlisted, key=lambda x: val_result[x]["precision_at_5"])
                if shortlisted else "classifier")
    print(f"Refitting selected {selected}", flush=True)
    fit_users = np.concatenate((train_users, validation_users))
    fit_frame = candidates.filter(pl.col("user_id").is_in(fit_users))
    final = _fit(fit_frame, selected, rounds)
    test = candidates.filter(pl.col("user_id").is_in(test_users))
    test_sizes = features.target_sizes.filter(pl.col("user_id").is_in(test_users))
    test_metrics = ranking_metrics(test, test_sizes, final.predict(test), 5)
    report = {
        "population": "same 10k original-training-user sample as model benchmark",
        "feature_columns": FEATURE_COLUMNS,
        "split_users": {"train": len(train_users), "validation": len(validation_users),
                        "test": len(test_users)},
        "rounds": rounds, "validation": val_result,
        "promotion_rule": "+0.005 absolute validation P@5 and positive paired interval",
        "selected": selected, "selected_test": test_metrics,
        "ranker_scores_are_probabilities": False,
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
                        default=Path("artifacts/ranking_objective_10k"))
    parser.add_argument("--max-users", type=int, default=10000)
    parser.add_argument("--rounds", type=int, default=350)
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.max_users, args.rounds)
