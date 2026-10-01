"""Full-population, GPU-accelerated, one-factor-at-a-time XGBoost sweep.

Every challenger uses 600 rounds and the same user split/features. Configs are
selected on the round-selection validation users. Only the validation winner
is refit on train plus selection users and scored on the historical test split.
"""

import argparse
import gc
import json
import os
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import psutil
import xgboost
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split

from grocery_recommender.data import load_bundle, prepare_data, validate_bundle
from grocery_recommender.features import FEATURE_COLUMNS, build_features
from grocery_recommender.full_train import _matrix
from grocery_recommender.metrics import rank_candidates, ranking_metrics
from grocery_recommender.modeling import new_scorer, split_user_ids


BASE_PARAMS = {
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "max_bin": 256,
}

CONFIGS = {
    "gpu_baseline": {},
    "depth_8": {"max_depth": 8},
    "depth_10": {"max_depth": 10},
    "min_child_weight_5": {"min_child_weight": 5},
    "colsample_1_0": {"colsample_bytree": 1.0},
    "subsample_1_0": {"subsample": 1.0},
    "reg_lambda_5": {"reg_lambda": 5.0},
    "max_bin_512": {"max_bin": 512},
}


def _build_shards(data_dir: Path, prepared, users: np.ndarray,
                  split_name: str, temp: Path, batch_size: int,
                  validate: bool = True) -> tuple[list[tuple[Path, int]], list[pl.DataFrame], float, int]:
    shards: list[tuple[Path, int]] = []
    sizes: list[pl.DataFrame] = []
    started = time.perf_counter()
    peak_rss = psutil.Process().memory_info().rss
    for index, start in enumerate(range(0, len(users), batch_size)):
        user_batch = users[start:start + batch_size].tolist()
        bundle = load_bundle(data_dir, user_ids=user_batch, prepared=prepared)
        if validate:
            validate_bundle(bundle)
        features = build_features(bundle)
        frame = features.candidates.sort("user_id", "product_id")
        shard = temp / f"{split_name}_{index:03d}.parquet"
        frame.write_parquet(shard)
        shards.append((shard, frame.height))
        sizes.append(features.target_sizes)
        peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
        print(f"{split_name} batch {index + 1}: {len(user_batch)} users, "
              f"{frame.height} pairs, RSS={psutil.Process().memory_info().rss / 2**30:.2f} GiB",
              flush=True)
        del frame, features, bundle
    return shards, sizes, time.perf_counter() - started, peak_rss


def _read(shards: list[tuple[Path, int]]) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(path) for path, _ in shards])


def _metrics(model, validation: pl.DataFrame, sizes: pl.DataFrame) -> tuple[dict, np.ndarray]:
    scores = model.predict(validation)
    result = ranking_metrics(validation, sizes, scores, 5)
    labels = validation["label"].to_numpy()
    result["average_precision"] = float(average_precision_score(labels, scores))
    result["brier"] = float(brier_score_loss(labels, scores))
    return result, scores


def _display_metrics(frame: pl.DataFrame, scores: np.ndarray,
                     target_sizes: pl.DataFrame) -> dict:
    top = rank_candidates(frame.select("user_id", "product_id", "label"), scores)
    fixed = ranking_metrics(frame, target_sizes, scores, 5)
    shown = top.filter((pl.col("rank") <= 5) & (pl.col("score") >= 0.5))
    shown_count = shown.height
    hit_count = int(shown["label"].sum()) if shown_count else 0
    users = target_sizes.height
    repeat_items = int(frame["label"].sum())
    precision = hit_count / shown_count if shown_count else 0.0
    recall = hit_count / repeat_items if repeat_items else 0.0
    return {
        "fixed_top_5": fixed,
        "threshold_0_5": {
            "suggestions_shown": shown_count,
            "matching_suggestions": hit_count,
            "precision_among_shown": precision,
            "mean_shown_per_customer": shown_count / users,
            "share_with_no_suggestions": 1.0 - shown["user_id"].n_unique() / users,
            "repeat_item_recall_micro": recall,
            "f1_micro": 2 * precision * recall / (precision + recall)
            if precision + recall else 0.0,
        },
    }


def run(data_dir: Path, output_dir: Path, rounds: int = 600,
        batch_size: int = 7500, n_jobs: int = 12) -> dict:
    if rounds < 1 or batch_size < 1 or n_jobs < 1:
        raise ValueError("rounds, batch_size and n_jobs must be positive")
    build = xgboost.build_info()
    if not build.get("USE_CUDA"):
        raise RuntimeError("This XGBoost build does not include CUDA support")
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_data(data_dir)
    labeled_users = prepared.orders.filter(pl.col("eval_set") == "train")[
        "user_id"
    ].to_numpy()
    train_users, validation_users, test_users = split_user_ids(labeled_users)
    selection_users, decision_users = train_test_split(
        np.sort(validation_users), test_size=0.5, random_state=42,
    )
    memory_before = psutil.virtual_memory()
    process = psutil.Process()
    start_run = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="capacity_", dir=output_dir) as temp_name:
        temp = Path(temp_name)
        profile_users = train_users[:15000]
        batch_profile = {}
        for profile_size in dict.fromkeys((5000, batch_size)):
            profile_dir = temp / f"batch_profile_{profile_size}"
            profile_dir.mkdir()
            _, _, profile_seconds, profile_peak_rss = _build_shards(
                data_dir, prepared, profile_users, f"profile_{profile_size}",
                profile_dir, profile_size, validate=False,
            )
            batch_profile[str(profile_size)] = {
                "customers_profiled": len(profile_users),
                "feature_generation_seconds": profile_seconds,
                "peak_process_rss_bytes": profile_peak_rss,
            }
            print(f"Batch profile {profile_size}: {profile_seconds:.1f}s, "
                  f"peak RSS={profile_peak_rss / 2**30:.2f} GiB", flush=True)
        train_shards, _, train_prep_seconds, peak_rss_train = _build_shards(
            data_dir, prepared, train_users, "train", temp, batch_size,
        )
        val_shards, val_size_parts, val_prep_seconds, peak_rss_val = _build_shards(
            data_dir, prepared, selection_users, "validation", temp, batch_size,
        )
        train_x, train_y = _matrix(train_shards, temp / "train")
        validation = _read(val_shards)
        selection_sizes = pl.concat(val_size_parts)
        train_rows = len(train_y)
        results = {}
        best_name = None
        best_p5 = float("-inf")
        best_config = None

        for name, overrides in CONFIGS.items():
            params = {**BASE_PARAMS, **overrides}
            print(f"Fitting {name}: {params}, {rounds} rounds, CUDA, "
                  f"n_jobs={n_jobs} on {train_rows:,} rows", flush=True)
            scorer = new_scorer("xgboost")
            scorer.estimator.set_params(
                n_estimators=rounds, **params, tree_method="hist",
                device="cuda", n_jobs=n_jobs, verbosity=0,
            )
            fit_start = time.perf_counter()
            scorer.estimator.fit(train_x, train_y)
            fit_seconds = time.perf_counter() - fit_start
            predict_start = time.perf_counter()
            metrics, scores = _metrics(scorer, validation, selection_sizes)
            predict_seconds = time.perf_counter() - predict_start
            raw_model = scorer.estimator.get_booster().save_raw(raw_format="ubj")
            results[name] = {
                "parameter_overrides": overrides,
                "parameters": params,
                "rounds": rounds,
                "validation": metrics,
                "fit_seconds": fit_seconds,
                "validation_predict_seconds": predict_seconds,
                "serialized_model_bytes": len(raw_model),
                "paired_users": len(selection_users),
            }
            (output_dir / "validation_progress.json").write_text(
                json.dumps({
                    "rounds": rounds,
                    "batch_size_users": batch_size,
                    "completed_configurations": results,
                }, indent=2) + "\n", encoding="utf-8",
            )
            p5 = metrics["precision_at_5"]
            print(f"{name}: validation P@5={p5:.5f}, "
                  f"AP={metrics['average_precision']:.5f}, "
                  f"fit={fit_seconds:.1f}s, model={len(raw_model) / 2**20:.1f} MiB",
                  flush=True)
            if p5 > best_p5:
                best_name, best_p5, best_config = name, p5, params
            del scorer, scores, raw_model
            gc.collect()

        best_config = {**best_config, "n_estimators": rounds,
                       "tree_method": "hist", "device": "cuda",
                       "n_jobs": n_jobs, "verbosity": 0}
        print(f"Validation winner: {best_name}; refitting on train plus round-selection "
              "users before test scoring", flush=True)
        selection_shard = temp / "selected_validation.parquet"
        validation.write_parquet(selection_shard)
        refit_x, refit_y = _matrix(
            train_shards + [(selection_shard, validation.height)], temp / "refit",
        )
        winner = new_scorer("xgboost")
        winner.estimator.set_params(**best_config)
        winner.estimator.fit(refit_x, refit_y)
        joblib.dump(winner, output_dir / "evaluated_model.joblib")

        test_shards, test_size_parts, _, _ = _build_shards(
            data_dir, prepared, test_users, "test", temp, batch_size,
        )
        test = _read(test_shards)
        test_sizes = pl.concat(test_size_parts)
        test_metrics, test_scores = _metrics(winner, test, test_sizes)
        test_display = _display_metrics(test, test_scores, test_sizes)
        joblib.dump(winner, output_dir / "model.joblib")

        result = {
            "rounds": rounds,
            "candidate_rows": {"train": train_rows, "selection_validation": validation.height,
                               "test": test.height},
            "users": {"train": len(train_users), "selection_validation": len(selection_users),
                      "decision_validation": len(decision_users), "test": len(test_users)},
            "hardware": {
                "logical_cpus": os.cpu_count(),
                "xgboost_n_jobs": n_jobs,
                "xgboost_device": "cuda",
                "xgboost_version": xgboost.__version__,
                "cuda_build": build.get("CUDA_VERSION"),
                "gpu_name": "NVIDIA GeForce RTX 3060 Laptop GPU",
                "gpu_vram_bytes": 6144 * 1024 * 1024,
                "system_memory_bytes": memory_before.total,
                "available_memory_at_start_bytes": memory_before.available,
                "peak_process_rss_bytes_during_preprocessing": max(peak_rss_train,
                                                                    peak_rss_val),
            },
            "batch_size_users": batch_size,
            "batch_size_profile": batch_profile,
            "preprocessing_seconds": {"train": train_prep_seconds,
                                      "selection_validation": val_prep_seconds},
            "configurations": results,
            "selected_on_validation": best_name,
            "selected_parameters": best_config,
            "selected_test_candidate_metrics": test_metrics,
            "selected_test_display_metrics": test_display,
            "test_status": (
                "Historical benchmark: the original test split has been inspected "
                "during prior project analysis."
            ),
            "elapsed_seconds": time.perf_counter() - start_run,
        }
        (output_dir / "metrics.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8",
        )
        print(json.dumps(result, indent=2), flush=True)
        del train_x, train_y, refit_x, refit_y, validation, test, test_scores
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/capacity_gpu_600"))
    parser.add_argument("--rounds", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=7500,
                        help="customers processed per feature-generation batch")
    parser.add_argument("--n-jobs", type=int, default=12)
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.rounds, args.batch_size, args.n_jobs)
