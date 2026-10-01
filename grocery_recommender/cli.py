"""Command line entry points for audit, training, and scoring."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import polars as pl

from .augmentation import historical_training_examples
from .basket_cli import basket_recommend
from .data import load_bundle, source_profile, validate_bundle
from .discovery_preview import preview_candidates, run_preview
from .features import FEATURE_COLUMNS, REPLENISHMENT_PILOT_COLUMNS, build_features
from .full_train import run_full_train
from .hybrid import combine_recommendations
from .metrics import rank_candidates, ranking_metrics
from .modeling import (
    LEGACY_FEATURE_COLUMNS, evaluate_scorer, fit_scorer, new_scorer, split_user_ids,
)


MODEL_NAMES = ("logistic", "random_forest", "xgboost", "xgboost_ranker", "catboost")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _distribution(series: pl.Series) -> dict[str, float]:
    return {
        "median": float(series.quantile(0.5)),
        "p90": float(series.quantile(0.9)),
        "p99": float(series.quantile(0.99)),
    }


def run_audit(args: argparse.Namespace) -> None:
    bundle = load_bundle(args.data_dir, max_users=args.max_users)
    quality = validate_bundle(bundle)
    feature_set = build_features(bundle)
    quality.update({
        "candidate_pairs": feature_set.candidates.height,
        "candidate_hits": feature_set.candidate_hits,
        "candidate_recall": feature_set.candidate_hits / feature_set.target_item_count,
        "target_reorder_rate": feature_set.candidate_hits / feature_set.target_item_count,
        "source": source_profile(args.data_dir),
        "exploration": {
            "historical_orders_per_user": _distribution(
                bundle.history_orders.group_by("user_id").len()["len"]
            ),
            "historical_basket_size": _distribution(
                bundle.history_lines.group_by("order_id").len()["len"]
            ),
            "target_basket_size": _distribution(
                bundle.target_lines.group_by("order_id").len()["len"]
            ),
            "reorder_candidates_per_user": _distribution(
                feature_set.candidates.group_by("user_id").len()["len"]
            ),
            "share_of_users_with_a_reorder_in_next_basket":
                feature_set.candidates.filter(pl.col("label") == 1)
                .select(pl.col("user_id").n_unique()).item() / bundle.target_orders.height,
        },
    })
    _write_json(args.output_dir / "data_audit.json", quality)
    print(json.dumps(quality, indent=2))


def run_train(args: argparse.Namespace) -> None:
    bundle = load_bundle(args.data_dir, max_users=args.max_users)
    quality = validate_bundle(bundle)
    feature_set = build_features(bundle)
    candidates = feature_set.candidates.sort("user_id", "product_id")
    train_users, validation_users, test_users = split_user_ids(
        feature_set.target_sizes["user_id"].to_numpy()
    )
    train = candidates.filter(pl.col("user_id").is_in(train_users))
    validation = candidates.filter(pl.col("user_id").is_in(validation_users))
    test = candidates.filter(pl.col("user_id").is_in(test_users))
    validation_sizes = feature_set.target_sizes.filter(pl.col("user_id").is_in(validation_users))
    test_sizes = feature_set.target_sizes.filter(pl.col("user_id").is_in(test_users))
    if validation.is_empty() or train.is_empty() or test.is_empty():
        raise ValueError("Empty training, validation, or test candidates")
    original_train_pairs = train.height
    if args.history_snapshots:
        if args.feature_set != "current_no_global":
            raise ValueError("Historical snapshots require --feature-set current_no_global")
        train = train.with_columns(
            (pl.col("user_order_count") + 1).alias("snapshot_id")
        )
        historical = historical_training_examples(bundle, train_users, args.history_snapshots)
        train = pl.concat([train, historical.select(train.columns)], how="vertical")

    requested = list(dict.fromkeys(args.models))
    if any(name not in MODEL_NAMES for name in requested):
        raise ValueError(f"Models must be drawn from: {', '.join(MODEL_NAMES)}")
    validation_results = {}
    if args.feature_set == "legacy":
        model_features = LEGACY_FEATURE_COLUMNS
    elif args.feature_set == "current_no_global":
        model_features = tuple(
            name for name in FEATURE_COLUMNS
            if name not in {"product_purchase_count", "product_reorder_rate"}
        )
    else:
        model_features = tuple(FEATURE_COLUMNS)
    best_name = None
    best_precision = float("-inf")
    for name in ["frequency", *requested]:
        print(f"Fitting and evaluating {name}...", flush=True)
        scorer = new_scorer(name)
        scorer.feature_columns = model_features
        scorer = fit_scorer(scorer, train)
        metrics, scores = evaluate_scorer(scorer, validation, validation_sizes, args.k)
        validation_results[name] = metrics
        print(json.dumps({name: metrics}, indent=2), flush=True)
        precision = metrics[f"precision_at_{args.k}"]
        if precision > best_precision:
            best_name, best_precision = name, precision

    print(f"Refitting selected model {best_name} on train + validation users...", flush=True)
    final_validation = (
        validation.with_columns((pl.col("user_order_count") + 1).alias("snapshot_id"))
        if args.history_snapshots else validation
    )
    final_train = pl.concat([train, final_validation])
    if args.history_snapshots:
        validation_history = historical_training_examples(
            bundle, validation_users, args.history_snapshots
        )
        final_train = pl.concat(
            [final_train, validation_history.select(final_train.columns)], how="vertical"
        )
    best_scorer = new_scorer(best_name)
    best_scorer.feature_columns = model_features
    best_scorer = fit_scorer(best_scorer, final_train)
    test_metrics, test_scores = evaluate_scorer(best_scorer, test, test_sizes, args.k)
    _, frequency_test_scores = evaluate_scorer(
        new_scorer("frequency"), test, test_sizes, args.k
    )
    cutoffs = sorted({1, 3, 5, 10, args.k})
    test_ranking_by_k = {
        str(cutoff): ranking_metrics(test, test_sizes, test_scores, cutoff)
        for cutoff in cutoffs
    }
    test_frequency_by_k = {
        str(cutoff): ranking_metrics(test, test_sizes, frequency_test_scores, cutoff)
        for cutoff in cutoffs
    }
    oracle_scores = test["label"].to_numpy().astype(float)
    test_oracle_by_k = {
        str(cutoff): ranking_metrics(test, test_sizes, oracle_scores, cutoff)
        for cutoff in cutoffs
    }
    test_hits = int(test.select(pl.col("label").sum()).item())
    test_items = int(test_sizes.select(pl.col("target_size").sum()).item())
    summary = {
        "best_model": best_name,
        "selection_metric": f"precision_at_{args.k}",
        "recommendation_timing": "before_next_order_starts",
        "k": args.k,
        "max_users": args.max_users,
        "train_users": len(train_users),
        "validation_users": len(validation_users),
        "test_users": len(test_users),
        "train_candidate_pairs": train.height,
        "original_train_candidate_pairs": original_train_pairs,
        "history_snapshots": args.history_snapshots,
        "validation_candidate_pairs": validation.height,
        "test_candidate_pairs": test.height,
        "test_candidate_recall": test_hits / test_items,
        "test_positive_rate": test_hits / test.height,
        "feature_set": args.feature_set,
        "feature_columns": model_features,
        "validation_models": validation_results,
        "test_selected_model": test_metrics,
        "test_ranking_by_k": test_ranking_by_k,
        "test_frequency_by_k": test_frequency_by_k,
        "test_oracle_by_k": test_oracle_by_k,
        "data_quality": quality,
        "note": "Candidates are previously purchased products; ranking recall uses the full next basket.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_scorer, args.output_dir / "model.joblib")
    _write_json(args.output_dir / "metrics.json", summary)
    ranked = rank_candidates(
        test.select("user_id", "product_id", "product_name", "label"), test_scores
    ).filter(pl.col("rank") <= args.k)
    ranked.select("user_id", "product_id", "product_name", "rank", "score", "label").write_csv(
        args.output_dir / "test_recommendations.csv"
    )
    print(json.dumps({"selected_test_result": test_metrics}, indent=2), flush=True)
    print(f"Saved model, metrics, and test recommendations to {args.output_dir}")


def run_recommend(args: argparse.Namespace) -> None:
    if args.k < 1:
        raise ValueError("--k must be positive")
    model_path = args.output_dir / "model.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Train a model first: {model_path}")
    scorer = joblib.load(model_path)
    if args.min_probability is not None:
        if not 0 <= args.min_probability <= 1:
            raise ValueError("--min-probability must be between 0 and 1")
        if scorer.name in ("frequency", "xgboost_ranker"):
            raise ValueError("--min-probability requires a probability model")
    bundle = load_bundle(args.data_dir, user_id=args.user_id)
    validate_bundle(bundle)
    needs_replenishment = any(name in scorer.feature_columns
                              for name in REPLENISHMENT_PILOT_COLUMNS)
    candidates = build_features(
        bundle, include_replenishment_pilot=needs_replenishment,
    ).candidates
    scores = scorer.predict(candidates)
    ranked = rank_candidates(
        candidates.select("user_id", "product_id", "product_name"), scores
    )
    cutoff = args.min_probability if args.min_probability is not None else 0.0
    accepted = ranked.filter((pl.col("rank") <= args.k) & (pl.col("score") >= cutoff))
    discovery_count = min(args.max_discovery, max(0, args.k - accepted.height))
    discovery = pl.DataFrame(schema={
        "user_id": pl.Int64, "product_id": pl.Int64,
        "product_name": pl.String, "preview_rank": pl.UInt32,
    })
    if discovery_count:
        discovery, report = preview_candidates(
            bundle, args.discovery_dir, args.discovery_ranker_dir, discovery_count,
        )
        if report["ranking"] != "specialist_model":
            raise ValueError(
                "Hybrid recommendations need a compatible discovery specialist. "
                "Run python -m experiments.discovery_ranker with the selected "
                "retrieval artifacts, or use --max-discovery 0."
            )
    display = combine_recommendations(ranked, discovery, args.k, cutoff,
                                      args.max_discovery)
    if display.is_empty():
        print("No recommendation candidates available at the requested cutoff.")
    else:
        print(f"Repeat suggestions meet the {cutoff:.2f} reorder probability cutoff. "
              "Discovery suggestions are products new to you.")
        with pl.Config(tbl_width_chars=140, fmt_str_lengths=45):
            print(display.select("rank", pl.col("recommendation_type").alias("type"),
                                 "product_id", "product_name",
                                 pl.col("estimated_reorder_probability")
                                 .alias("reorder_probability")))


def run_predict_reorders(args: argparse.Namespace) -> None:
    decision_path = args.output_dir / "decision.json"
    model_path = args.output_dir / "model.joblib"
    if not decision_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(
            f"Run train-full to create a validated reorder cutoff in {args.output_dir}"
        )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    threshold = float(decision["threshold"])
    scorer = joblib.load(model_path)
    bundle = load_bundle(args.data_dir, user_id=args.user_id)
    validate_bundle(bundle)
    needs_replenishment = any(name in scorer.feature_columns
                              for name in REPLENISHMENT_PILOT_COLUMNS)
    candidates = build_features(
        bundle, include_replenishment_pilot=needs_replenishment,
    ).candidates
    ranked = rank_candidates(
        candidates.select("user_id", "product_id", "product_name"),
        scorer.predict(candidates),
    ).filter(pl.col("score") >= threshold)
    print(f"Predicted reorders for user {args.user_id} at cutoff {threshold:.2f}:")
    if ranked.is_empty():
        print("No products passed the validated cutoff.")
    else:
        score_label = ("model_score" if scorer.name in ("frequency", "xgboost_ranker")
                       else "estimated_probability")
        print(ranked.select("rank", "product_id", "product_name",
                            pl.col("score").alias(score_label)))


def run_basket_recommend(args: argparse.Namespace) -> None:
    mode = "session" if args.command == "recommend-session" else "pre_order"
    ranked = basket_recommend(
        args.data_dir, args.output_dir, args.user_id, mode,
        cart_product_ids=getattr(args, "cart_product_ids", None),
        preview_new=args.preview_new, k=args.k,
    )
    if args.preview_new:
        preview_path = args.output_dir / "preview_metrics.json"
        if preview_path.is_file():
            preview = json.loads(preview_path.read_text(encoding="utf-8"))
            precision = preview["new_only_test"]["precision_at_5"]
            print(f"Experimental new-product-only preview; pilot-test "
                  f"Precision@5={precision:.3f}.")
        else:
            print("Experimental new-product-only preview; no held-out preview metrics found.")
    else:
        report = json.loads((args.output_dir / "metrics.json").read_text(encoding="utf-8"))
        print(f"Pilot-selected model: {report['selected']}")
    if ranked.is_empty():
        print("No candidate products available.")
    else:
        print(ranked)


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Instacart next-basket reorder recommender")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "train", "train-full", "recommend",
                    "predict-reorders", "recommend-session", "recommend-preorder-pilot",
                    "recommend-discovery-preview"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--data-dir", type=Path, default=Path("."))
        default_output = (
            Path("artifacts/full_v2") if command in
            ("train-full", "recommend", "predict-reorders") else
            Path("artifacts/session_pilot") if command == "recommend-session" else
            Path("artifacts/preorder_pilot") if command == "recommend-preorder-pilot" else
            Path("artifacts/discovery_10k_ablation") if command == "recommend-discovery-preview" else
            Path("artifacts")
        )
        cmd.add_argument("--output-dir", type=Path, default=default_output)
        if command in ("audit", "train"):
            cmd.add_argument("--max-users", type=int, default=None)
        if command == "train":
            cmd.add_argument("--models", nargs="+", default=["logistic", "xgboost"])
            cmd.add_argument("--k", type=int, default=5)
            cmd.add_argument("--feature-set", choices=["legacy", "current", "current_no_global"], default="current")
            cmd.add_argument("--history-snapshots", type=int, default=0)
        if command == "recommend":
            cmd.add_argument("--user-id", type=int, required=True)
            cmd.add_argument("--k", type=int, default=5)
            cmd.add_argument(
                "--min-probability", type=float, default=0.5,
                help="minimum estimated reorder probability (default: 0.5; use 0 for an unfiltered top five)",
            )
            cmd.add_argument(
                "--max-discovery", type=int, choices=(0, 1, 2), default=2,
                help="maximum Discovery suggestions in remaining display slots (default: 2)",
            )
            cmd.add_argument("--discovery-dir", type=Path,
                             default=Path("artifacts/discovery_10k_ablation"))
            cmd.add_argument("--discovery-ranker-dir", type=Path,
                             default=Path("artifacts/discovery_ranker_filled_10k"))
        if command == "predict-reorders":
            cmd.add_argument("--user-id", type=int, required=True)
        if command == "recommend-discovery-preview":
            cmd.add_argument("--user-id", type=int, required=True)
            cmd.add_argument("--k", type=int, default=5)
            cmd.add_argument("--ranker-dir", type=Path,
                             default=Path("artifacts/discovery_ranker_filled_10k"))
        if command in ("recommend-session", "recommend-preorder-pilot"):
            cmd.add_argument("--user-id", type=int, required=True)
            cmd.add_argument("--k", type=int, default=5)
            cmd.add_argument("--preview-new", action="store_true")
            if command == "recommend-session":
                cmd.add_argument("--cart-product-ids", nargs=2, type=int,
                                 required=True)
        if command == "train-full":
            cmd.add_argument("--batch-size", type=int, default=5000)
            cmd.add_argument("--max-users", type=int, default=None)
            cmd.add_argument("--rounds", type=int, nargs="+", default=[150, 250, 400, 600])
    args = parser.parse_args(argv)
    if args.command == "audit":
        run_audit(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "train-full":
        run_full_train(args.data_dir, args.output_dir, args.batch_size,
                       tuple(args.rounds), args.max_users)
    elif args.command == "recommend":
        run_recommend(args)
    elif args.command == "recommend-discovery-preview":
        run_preview(args.data_dir, args.output_dir, args.user_id, args.k,
                    args.ranker_dir)
    elif args.command in ("recommend-session", "recommend-preorder-pilot"):
        run_basket_recommend(args)
    else:
        run_predict_reorders(args)


if __name__ == "__main__":
    main()
