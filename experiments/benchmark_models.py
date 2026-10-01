"""User-disjoint comparison of the original five modeling choices.

The standard variant uses identical features for every model. The enriched
variant is an explicit feature ablation with product priors and native
CatBoost categories. Both sample only from the original full-run training
population; the historical full test is not used for model selection.
"""

import argparse
import gc
import importlib.metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score)

from grocery_recommender.data import load_bundle, prepare_data, validate_bundle
from grocery_recommender.features import (FEATURE_COLUMNS,
                                          REPLENISHMENT_PILOT_COLUMNS,
                                          build_features)
from grocery_recommender.metrics import rank_candidates, ranking_metrics
from grocery_recommender.modeling import fit_scorer, new_scorer, split_user_ids
from grocery_recommender.population_features import (PRODUCT_PRIOR_COLUMNS,
                                                     fit_product_priors)


def _hits(frame: pl.DataFrame, scores: np.ndarray, users: np.ndarray) -> np.ndarray:
    ranked = rank_candidates(frame.select("user_id", "product_id", "label"), scores)
    counts = ranked.filter(pl.col("rank") <= 5).group_by("user_id").agg(
        pl.col("label").sum().alias("hits")
    )
    return (pl.DataFrame({"user_id": users}).join(counts, on="user_id", how="left",
                                             maintain_order="left")["hits"]
            .fill_null(0).to_numpy())


def _paired_interval(a: np.ndarray, b: np.ndarray) -> list[float]:
    differences = (a - b) / 5
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(differences), size=(1200, len(differences)))
    return [float(x) for x in np.quantile(differences[indices].mean(axis=1),
                                          [0.025, 0.975])]


def _evaluate(model, frame: pl.DataFrame, sizes: pl.DataFrame) -> tuple[dict, np.ndarray]:
    scores = model.predict(frame)
    metrics = {str(k): ranking_metrics(frame, sizes, scores, k)
               for k in (1, 3, 5, 10)}
    labels = frame["label"].to_numpy()
    metrics["average_precision"] = float(average_precision_score(labels, scores))
    if model.name != "xgboost_ranker":
        predicted = scores >= 0.5
        metrics["candidate_classification_at_0_5"] = {
            "precision": float(precision_score(labels, predicted, zero_division=0)),
            "recall": float(recall_score(labels, predicted, zero_division=0)),
            "f1": float(f1_score(labels, predicted, zero_division=0)),
            "predicted_positive_products": int(predicted.sum()),
            "actual_positive_products": int(labels.sum()),
        }
    if model.name not in {"frequency", "xgboost_ranker"}:
        metrics["brier"] = float(brier_score_loss(labels, scores))
    return metrics, scores


def run(data_dir: Path, output_dir: Path, max_users: int = 10000,
        variant: str = "standard", rounds: int = 350) -> dict:
    if max_users < 100 or variant not in {"standard", "enriched"} or rounds < 1:
        raise ValueError("Use at least 100 users, a valid variant and positive rounds")
    prepared = prepare_data(data_dir)
    all_labeled = prepared.orders.filter(pl.col("eval_set") == "train")["user_id"].to_numpy()
    original_train = split_user_ids(all_labeled)[0]
    sampled = np.random.default_rng(2026).choice(
        original_train, size=min(max_users, len(original_train)), replace=False,
    )
    train_users, validation_users, test_users = split_user_ids(sampled)
    print(f"Loading {len(sampled)} users; variant={variant}", flush=True)
    bundle = load_bundle(data_dir, user_ids=sampled.tolist(), prepared=prepared)
    validate_bundle(bundle)
    rich = variant == "enriched"
    features = build_features(bundle, include_replenishment_pilot=rich)
    frame = features.candidates.sort("user_id", "product_id")
    train = frame.filter(pl.col("user_id").is_in(train_users))
    validation = frame.filter(pl.col("user_id").is_in(validation_users))
    test = frame.filter(pl.col("user_id").is_in(test_users))
    val_sizes = features.target_sizes.filter(pl.col("user_id").is_in(validation_users))
    test_sizes = features.target_sizes.filter(pl.col("user_id").is_in(test_users))
    if rich:
        columns = tuple(FEATURE_COLUMNS + REPLENISHMENT_PILOT_COLUMNS +
                        list(PRODUCT_PRIOR_COLUMNS))
        categories = ("product_id", "aisle_id", "department_id")
        priors = fit_product_priors(bundle, train_users)
        names = ("frequency", "xgboost", "catboost_identity")
    else:
        columns, categories, priors = tuple(FEATURE_COLUMNS), (), None
        names = ("frequency", "logistic", "random_forest", "xgboost", "catboost")
    results = {}
    best_name, best_precision = None, -1.0
    baseline_hits = None

    def make_model(name: str, population: pl.DataFrame | None):
        scorer = new_scorer("catboost" if name == "catboost_identity" else name)
        scorer.feature_columns = columns + (categories if name == "catboost_identity" else ())
        scorer.categorical_columns = categories if name == "catboost_identity" else ()
        scorer.population_stats = population
        if scorer.name == "xgboost":
            scorer.estimator.set_params(n_estimators=rounds)
        if scorer.name == "catboost":
            scorer.estimator.set_params(iterations=rounds)
        return scorer

    for name in names:
        print(f"Fitting {name} on {len(train_users)} users...", flush=True)
        scorer = fit_scorer(make_model(name, priors), train)
        val_metrics, val_scores = _evaluate(scorer, validation, val_sizes)
        hits = _hits(validation, val_scores, validation_users)
        if name == "frequency":
            baseline_hits = hits
        val_metrics["paired_p5_difference_vs_frequency"] = float(
            (hits - baseline_hits).mean() / 5
        )
        val_metrics["paired_p5_interval_vs_frequency"] = _paired_interval(
            hits, baseline_hits
        )
        results[name] = val_metrics
        score = val_metrics["5"]["precision_at_5"]
        print(f"{name} validation P@5={score:.5f}", flush=True)
        if score > best_precision:
            best_name, best_precision = name, score
        del scorer, val_scores
        gc.collect()

    print(f"Refitting validation winner: {best_name}", flush=True)
    final_users = np.concatenate([train_users, validation_users])
    final_priors = fit_product_priors(bundle, final_users) if rich else None
    final = fit_scorer(make_model(best_name, final_priors),
                       pl.concat([train, validation]).sort("user_id", "product_id"))
    test_metrics, test_scores = _evaluate(final, test, test_sizes)
    freq_scores = new_scorer("frequency").predict(test)
    test_metrics["paired_p5_interval_vs_frequency"] = _paired_interval(
        _hits(test, test_scores, test_users), _hits(test, freq_scores, test_users)
    )
    test_metrics["frequency_p5"] = ranking_metrics(
        test, test_sizes, freq_scores, 5
    )["precision_at_5"]
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / "train_user_ids.csv", train_users, fmt="%d")
    np.savetxt(output_dir / "validation_user_ids.csv", validation_users, fmt="%d")
    np.savetxt(output_dir / "test_user_ids.csv", test_users, fmt="%d")
    joblib.dump(final, output_dir / "model.joblib")
    summary = {
        "variant": variant, "recommendation_timing": "before_next_order_starts",
        "selection_rule": "highest validation Precision@5 on the same user split",
        "source_population": "sample from original full-run training users",
        "split_users": {"train": len(train_users), "validation": len(validation_users),
                        "test": len(test_users)},
        "candidate_rows": {"train": train.height, "validation": validation.height,
                           "test": test.height},
        "feature_columns": list(columns), "categorical_columns": list(categories),
        "model_rounds": rounds, "validation_models": results,
        "classification_metric_note": (
            "Average precision is the PR-AUC summary used here. Precision, recall and F1 "
            "use the same fixed score cutoff 0.5 on complete candidate sets. The "
            "frequency score is a historical purchase rate, not a calibrated "
            "next-order probability; its cutoff metrics are descriptive."
        ),
        "selected": best_name, "selected_test": test_metrics,
        "test_candidate_recall": float(test["label"].sum() / test_sizes["target_size"].sum()),
        "versions": {"python": sys.version.split()[0], "polars": pl.__version__,
                     "numpy": np.__version__,
                     "scikit_learn": importlib.metadata.version("scikit-learn"),
                     "xgboost": importlib.metadata.version("xgboost"),
                     "catboost": importlib.metadata.version("catboost")},
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    ranked = rank_candidates(
        test.select("user_id", "product_id", "product_name", "label"), test_scores
    ).filter(pl.col("rank") <= 5)
    ranked.write_csv(output_dir / "test_recommendations.csv")
    print(json.dumps({"selected": best_name, "test": test_metrics}, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/model_benchmark")
    parser.add_argument("--max-users", type=int, default=10000)
    parser.add_argument("--variant", choices=["standard", "enriched"],
                        default="standard")
    parser.add_argument("--rounds", type=int, default=350)
    arguments = parser.parse_args()
    run(arguments.data_dir, arguments.output_dir, arguments.max_users,
        arguments.variant, arguments.rounds)
