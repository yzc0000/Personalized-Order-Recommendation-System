"""Memory-bounded feature generation and full-population XGBoost training."""

import gc
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from .data import load_bundle, prepare_data, validate_bundle
from .decisions import choose_threshold, decision_metrics, probability_metrics
from .features import FEATURE_COLUMNS, build_features
from .metrics import rank_candidates, ranking_metrics
from .modeling import new_scorer, split_user_ids


def _matrix(shards: list[tuple[Path, int]], path: Path) -> tuple[np.memmap, np.memmap]:
    rows = sum(count for _, count in shards)
    x = np.lib.format.open_memmap(path.with_suffix(".x.npy"), mode="w+", dtype="float32",
                                  shape=(rows, len(FEATURE_COLUMNS)))
    y = np.lib.format.open_memmap(path.with_suffix(".y.npy"), mode="w+", dtype="int8",
                                  shape=(rows,))
    offset = 0
    for shard, count in shards:
        frame = pl.read_parquet(shard, columns=[*FEATURE_COLUMNS, "label"])
        x[offset:offset + count] = frame.select(FEATURE_COLUMNS).to_numpy().astype(np.float32)
        y[offset:offset + count] = frame["label"].to_numpy()
        offset += count
    x.flush()
    y.flush()
    return x, y


def _read_split(shards: list[tuple[Path, int]]) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(path) for path, _ in shards])


def _json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_full_train(data_dir: Path, output_dir: Path, batch_size: int = 5000,
                   rounds: tuple[int, ...] = (150, 250, 400, 600),
                   max_users: int | None = None) -> None:
    """Evaluate on unseen users, then fit the deployment model on all labels."""
    if batch_size < 1 or not rounds or any(n < 1 for n in rounds):
        raise ValueError("batch_size and boosting rounds must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_data(data_dir)
    eligible_frame = prepared.orders.filter(pl.col("eval_set") == "train").select("user_id")
    if max_users is not None:
        eligible_frame = eligible_frame.sample(n=min(max_users, eligible_frame.height), seed=42)
    eligible = eligible_frame["user_id"].to_numpy()
    user_splits = dict(zip(("train", "validation", "test"), split_user_ids(eligible)))
    shards: dict[str, list[tuple[Path, int]]] = {key: [] for key in user_splits}
    sizes: dict[str, list[pl.DataFrame]] = {key: [] for key in user_splits}
    total_items = 0
    candidate_hits = 0
    with tempfile.TemporaryDirectory(prefix="features_", dir=output_dir) as temp_name:
        temp = Path(temp_name)
        for split, users in user_splits.items():
            for index, start in enumerate(range(0, len(users), batch_size)):
                batch_users = users[start:start + batch_size].tolist()
                bundle = load_bundle(data_dir, user_ids=batch_users, prepared=prepared)
                validate_bundle(bundle)
                feature_set = build_features(bundle)
                frame = feature_set.candidates.sort("user_id", "product_id")
                shard = temp / f"{split}_{index:03d}.parquet"
                frame.write_parquet(shard)
                shards[split].append((shard, frame.height))
                sizes[split].append(feature_set.target_sizes)
                total_items += feature_set.target_item_count
                candidate_hits += feature_set.candidate_hits
                print(f"{split} batch {index + 1}: {len(batch_users)} users, "
                      f"{frame.height} candidates", flush=True)
                del frame, feature_set, bundle
        target_sizes = {split: pl.concat(parts) for split, parts in sizes.items()}
        train_x, train_y = _matrix(shards["train"], temp / "train")
        validation = _read_split(shards["validation"])
        selection_users, decision_users = train_test_split(
            np.sort(user_splits["validation"]), test_size=0.5, random_state=42
        )
        selection = validation.filter(pl.col("user_id").is_in(selection_users))
        decision = validation.filter(pl.col("user_id").is_in(decision_users))
        selection_sizes = target_sizes["validation"].filter(
            pl.col("user_id").is_in(selection_users)
        )
        selection_x = selection.select(FEATURE_COLUMNS).to_numpy().astype(np.float32)
        selection_y = selection["label"].to_numpy()
        scorer = new_scorer("xgboost")
        scorer.estimator.set_params(n_estimators=max(rounds))
        print(f"Fitting XGBoost for {max(rounds)} rounds on {len(train_y)} candidate rows...", flush=True)
        scorer.estimator.fit(train_x, train_y)
        frequency_scores = new_scorer("frequency").predict(selection)
        frequency_validation = ranking_metrics(selection, selection_sizes,
                                               frequency_scores, 5)
        round_results = {}
        best_round = None
        best_precision = float("-inf")
        for n in sorted(set(rounds)):
            scores = scorer.estimator.predict_proba(selection_x,
                                                   iteration_range=(0, n))[:, 1]
            metrics = ranking_metrics(selection, selection_sizes, scores, 5)
            metrics["pr_auc"] = float(average_precision_score(selection_y, scores))
            round_results[str(n)] = metrics
            print(f"Validation {n} rounds: Precision@5={metrics['precision_at_5']:.4f}",
                  flush=True)
            if metrics["precision_at_5"] > best_precision:
                best_round, best_precision = n, metrics["precision_at_5"]
        use_model = best_precision > frequency_validation["precision_at_5"]
        selected_name = "xgboost" if use_model else "frequency"
        print(f"Selected {selected_name}, {best_round if use_model else 0} rounds", flush=True)

        selection_shard = temp / "model_selection.parquet"
        selection.write_parquet(selection_shard)
        final_train_x, final_train_y = _matrix(
            shards["train"] + [(selection_shard, selection.height)], temp / "train_selection"
        )
        evaluated = new_scorer(selected_name)
        if use_model:
            evaluated.estimator.set_params(n_estimators=best_round)
            evaluated.estimator.fit(final_train_x, final_train_y)
        decision_scores = evaluated.predict(decision)
        threshold, threshold_trials = choose_threshold(
            decision["label"].to_numpy(), decision_scores, decision["user_id"].to_numpy()
        )
        validation_decision = decision_metrics(
            decision["label"].to_numpy(), decision_scores,
            decision["user_id"].to_numpy(), threshold,
        )
        validation_probability = (
            probability_metrics(decision["label"].to_numpy(), decision_scores)
            if use_model else None
        )
        print(f"Selected reorder cutoff {threshold:.2f} on {len(decision_users)} "
              f"separate decision-validation users", flush=True)
        test = _read_split(shards["test"])
        test_scores = evaluated.predict(test)
        test_decision = decision_metrics(
            test["label"].to_numpy(), test_scores, test["user_id"].to_numpy(), threshold
        )
        test_probability = (
            probability_metrics(test["label"].to_numpy(), test_scores)
            if use_model else None
        )
        test_metrics = {
            str(k): ranking_metrics(test, target_sizes["test"], test_scores, k)
            for k in (1, 3, 5, 10)
        }
        baseline_scores = new_scorer("frequency").predict(test)
        baseline_metrics = {
            str(k): ranking_metrics(test, target_sizes["test"], baseline_scores, k)
            for k in (1, 3, 5, 10)
        }
        oracle_scores = test["label"].to_numpy().astype(float)
        oracle_metrics = {
            str(k): ranking_metrics(test, target_sizes["test"], oracle_scores, k)
            for k in (1, 3, 5, 10)
        }
        test_items = int(target_sizes["test"].select(pl.col("target_size").sum()).item())
        test_hits = int(test.select(pl.col("label").sum()).item())
        joblib.dump(evaluated, output_dir / "evaluated_model.joblib")
        ranked_test = rank_candidates(
            test.select("user_id", "product_id", "product_name", "label"), test_scores
        )
        ranked_test.filter(pl.col("rank") <= 5).select(
            "user_id", "product_id", "product_name", "rank", "score", "label"
        ).write_csv(output_dir / "test_recommendations.csv")
        ranked_test.filter(pl.col("score") >= threshold).select(
            "user_id", "product_id", "product_name", "rank", "score", "label"
        ).write_csv(output_dir / "test_reorder_predictions.csv")
        _json(output_dir / "decision.json", {
            "threshold": threshold,
            "selection_metric": "micro_f0_5_on_reorder_candidates",
            "selection_users": len(decision_users),
            "evaluation_model": "evaluated_model.joblib",
            "deployment_model": "model.joblib",
            "label": "previously purchased product appears in the next order",
            "deployment_note": "The deployment model is refit on all labeled users after evaluation; "
                               "its cutoff comes from a separate evaluated model's validation users.",
        })
        summary = {
            "recommendation_timing": "before_next_order_starts",
            "selection_metric": "precision_at_5",
            "labeled_users": len(eligible),
            "split_users": {key: len(users) for key, users in user_splits.items()},
            "model_selection_users": len(selection_users),
            "decision_validation_users": len(decision_users),
            "split_candidate_pairs": {key: sum(count for _, count in parts)
                                      for key, parts in shards.items()},
            "feature_columns": FEATURE_COLUMNS,
            "feature_count": len(FEATURE_COLUMNS),
            "validation_frequency": frequency_validation,
            "validation_rounds": round_results,
            "validation_threshold_trials": threshold_trials,
            "validation_decision": validation_decision,
            "validation_probability": validation_probability,
            "selected_model": selected_name,
            "selected_rounds": best_round if use_model else 0,
            "decision_threshold": threshold,
            "test_selected_model": test_metrics,
            "test_decision": test_decision,
            "test_probability": test_probability,
            "test_frequency": baseline_metrics,
            "test_oracle": oracle_metrics,
            "test_candidate_recall": test_hits / test_items,
            "all_labeled_candidate_recall": candidate_hits / total_items,
            "test_model_path": "evaluated_model.joblib",
            "deployment_model_path": "model.joblib",
        }
        _json(output_dir / "metrics.json", summary)
        print(f"Held-out reorder precision={test_decision['precision']:.4f}, "
              f"recall={test_decision['recall_of_reorders']:.4f}; "
              f"Precision@5={test_metrics['5']['precision_at_5']:.4f}; "
              f"frequency={baseline_metrics['5']['precision_at_5']:.4f}; "
              f"oracle={oracle_metrics['5']['precision_at_5']:.4f}", flush=True)
        del scorer, evaluated, train_x, train_y, final_train_x, final_train_y
        del validation, selection, selection_x, selection_y, decision, test, ranked_test
        all_x, all_y = _matrix(
            shards["train"] + shards["validation"] + shards["test"], temp / "all_labeled"
        )
        deployment = new_scorer(selected_name)
        if use_model:
            deployment.estimator.set_params(n_estimators=best_round)
            print(f"Fitting deployment model on all {len(eligible)} labeled users...", flush=True)
            deployment.estimator.fit(all_x, all_y)
        joblib.dump(deployment, output_dir / "model.joblib")
        del all_x, all_y, deployment
        gc.collect()
    print(f"Saved evaluated and all-labeled models to {output_dir}", flush=True)
