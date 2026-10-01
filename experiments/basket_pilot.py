"""Evaluate co-basket discovery before or during the next shopping order."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from grocery_recommender.co_basket import (fit_co_basket_index,
                                           retrieve_co_basket_candidates)
from grocery_recommender.data import load_bundle, prepare_data
from grocery_recommender.features import FEATURE_COLUMNS, FeatureSet, build_features
from grocery_recommender.metrics import rank_candidates
from grocery_recommender.mixed import BASKET_FEATURE_COLUMNS, build_mixed_candidates
from grocery_recommender.modeling import fit_scorer, new_scorer, split_user_ids


def _targets(bundle, mode: str) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    target = bundle.target_lines.join(
        bundle.target_orders.select("order_id", "user_id"), on="order_id"
    )
    if mode == "pre_order":
        last = bundle.history_orders.group_by("user_id").agg(
            pl.col("order_number").max().alias("last_order_number")
        )
        last_orders = (
            bundle.history_orders.join(last, on="user_id")
            .filter(pl.col("order_number") == pl.col("last_order_number"))
            .select("order_id", "user_id")
        )
        anchors = bundle.history_lines.join(last_orders, on="order_id")
        sizes = target.group_by("user_id").agg(pl.len().alias("target_size"))
        return (anchors.select("user_id", "product_id"),
                target.select("user_id", "product_id"), sizes)
    if mode != "session":
        raise ValueError("mode must be pre_order or session")
    full_sizes = target.group_by("user_id").agg(pl.len().alias("full_basket_size"))
    eligible = full_sizes.filter(pl.col("full_basket_size") >= 2)
    target = target.join(eligible.select("user_id"), on="user_id")
    anchors = target.filter(pl.col("add_to_cart_order") <= 2)
    remaining = target.filter(pl.col("add_to_cart_order") > 2)
    sizes = eligible.with_columns(
        (pl.col("full_basket_size") - 2).alias("target_size")
    ).select("user_id", "target_size")
    if anchors.group_by("user_id").len().filter(pl.col("len") != 2).height:
        raise ValueError("Expected exactly two observed cart products per session")
    return (anchors.select("user_id", "product_id"),
            remaining.select("user_id", "product_id"), sizes)


def _metrics(frame: pl.DataFrame, sizes: pl.DataFrame, scores: np.ndarray) -> dict:
    ranked = rank_candidates(
        frame.select("user_id", "product_id", "label", "is_novel"), scores
    ).filter(pl.col("rank") <= 5)
    per_user = (
        sizes.join(ranked.group_by("user_id").agg(
            pl.col("label").sum().alias("hits"),
            (pl.col("label") * pl.col("is_novel")).sum().alias("novel_hits"),
            pl.col("is_novel").sum().alias("novel_shown"),
        ), on="user_id", how="left")
        .with_columns(
            pl.col("hits").fill_null(0),
            pl.col("novel_hits").fill_null(0),
            pl.col("novel_shown").fill_null(0),
        )
    )
    positives = per_user.filter(pl.col("target_size") > 0)
    return {
        "users": sizes.height,
        "users_with_no_remaining_items": sizes.filter(pl.col("target_size") == 0).height,
        "remaining_items": int(sizes["target_size"].sum()),
        "precision_at_5": float(per_user["hits"].mean() / 5),
        "recall_at_5_among_nonempty_targets": float(
            (positives["hits"] / positives["target_size"]).mean()
        ),
        "novel_items_in_top_five": int(per_user["novel_shown"].sum()),
        "novel_hits_in_top_five": int(per_user["novel_hits"].sum()),
    }


def _fit(frame: pl.DataFrame, users: np.ndarray, rounds: int,
         columns: tuple[str, ...]):
    train = frame.filter(pl.col("user_id").is_in(users))
    scorer = new_scorer("xgboost")
    scorer.feature_columns = columns
    scorer.estimator.set_params(n_estimators=rounds)
    return fit_scorer(scorer, train)


def run(data_dir: Path, output_dir: Path, mode: str = "session",
        max_users: int = 20000, top_products: int = 3000,
        novel_per_user: int = 40, rounds: int = 300) -> dict:
    prepared = prepare_data(data_dir)
    labeled = prepared.orders.filter(pl.col("eval_set") == "train")["user_id"].to_numpy()
    original_train = split_user_ids(labeled)[0]
    users = np.random.default_rng(42).choice(
        original_train, size=min(max_users, len(original_train)), replace=False,
    )
    train_users, validation_users, test_users = split_user_ids(users)
    print(f"Loading {len(users)} users...", flush=True)
    bundle = load_bundle(data_dir, user_ids=users.tolist(), prepared=prepared)
    base = build_features(bundle)
    anchors, remaining, sizes = _targets(bundle, mode)
    eligible = sizes["user_id"].to_numpy()
    base = FeatureSet(
        base.candidates.filter(pl.col("user_id").is_in(eligible)),
        sizes, remaining.height, 0,
    )
    print(f"Fitting order-level relationships on {len(train_users)} training histories...",
          flush=True)
    index = fit_co_basket_index(bundle, train_users, top_products=top_products)
    known = base.candidates.select("user_id", "product_id")
    print("Retrieving candidates...", flush=True)
    novel, known_scores = retrieve_co_basket_candidates(
        index, anchors, known, per_user=novel_per_user,
    )
    repeat, mixed = build_mixed_candidates(
        bundle, base, novel, known_scores, anchors, remaining,
        exclude_anchors=(mode == "session"),
    )
    split_sizes = {
        name: sizes.filter(pl.col("user_id").is_in(user_split))
        for name, user_split in (("validation", validation_users), ("test", test_users))
    }
    results = {}
    models = {}
    variants = (
        ("history", repeat, tuple(FEATURE_COLUMNS)),
        ("repeat", repeat, BASKET_FEATURE_COLUMNS),
        ("mixed", mixed, BASKET_FEATURE_COLUMNS),
    )
    for name, frame, columns in variants:
        print(f"Fitting {name} model on training users...", flush=True)
        models[name] = _fit(frame, train_users, rounds, columns)
        validation = frame.filter(pl.col("user_id").is_in(validation_users))
        results[name] = {
            "validation": _metrics(
                validation, split_sizes["validation"], models[name].predict(validation)
            )
        }
        print(f"{name} validation Precision@5: "
              f"{results[name]['validation']['precision_at_5']:.5f}", flush=True)
    repeat_gain = (results["repeat"]["validation"]["precision_at_5"] -
                   results["history"]["validation"]["precision_at_5"])
    selected = "repeat" if repeat_gain >= 0.005 else "history"
    mixed_gain = (results["mixed"]["validation"]["precision_at_5"] -
                  results[selected]["validation"]["precision_at_5"])
    if mixed_gain >= 0.005:
        selected = "mixed"
    print(f"Selected {selected}; repeat gain {repeat_gain:+.5f}; "
          f"mixed gain {mixed_gain:+.5f}", flush=True)
    selected_frame = mixed if selected == "mixed" else repeat
    selected_columns = (tuple(FEATURE_COLUMNS) if selected == "history"
                        else BASKET_FEATURE_COLUMNS)
    final_users = np.concatenate((train_users, validation_users))
    final_model = _fit(selected_frame, final_users, rounds, selected_columns)
    test = selected_frame.filter(pl.col("user_id").is_in(test_users))
    test_result = _metrics(test, split_sizes["test"], final_model.predict(test))
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(index, output_dir / "co_basket_index.joblib")
    joblib.dump(final_model, output_dir / "model.joblib")
    # The mixed training-only model supports a clearly experimental new-item
    # preview if mixed failed the promotion gate.
    if selected != "mixed":
        joblib.dump(models["mixed"], output_dir / "mixed_preview_model.joblib")
    report = {
        "mode": mode,
        "sample_users": len(users),
        "eligible_users": sizes.height,
        "split_users": {"train": len(train_users), "validation": len(validation_users),
                        "test": len(test_users)},
        "retrieval_fit_users": len(train_users),
        "retrieval_source": "prior orders of training users only",
        "top_products": len(index.product_ids),
        "novel_per_user": novel_per_user,
        "rounds": rounds,
        "repeat_candidate_recall": (float(repeat["label"].sum() / remaining.height)
                                    if remaining.height else None),
        "mixed_candidate_recall": (float(mixed["label"].sum() / remaining.height)
                                   if remaining.height else None),
        "novel_candidates": novel.height,
        "novel_candidate_hits": int(mixed.filter(pl.col("is_novel") == 1)["label"].sum()),
        "results": results,
        "repeat_validation_gain_over_history": repeat_gain,
        "mixed_validation_gain_over_chosen_baseline": mixed_gain,
        "minimum_material_gain": 0.005,
        "selected": selected,
        "selected_test": test_result,
        "test_population_note": (
            "session: users with at least two cart items, including baskets that end there"
            if mode == "session" else "pre-order: all sampled users"
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/session_pilot"))
    parser.add_argument("--mode", choices=["session", "pre_order"], default="session")
    parser.add_argument("--max-users", type=int, default=20000)
    parser.add_argument("--top-products", type=int, default=3000)
    parser.add_argument("--novel-per-user", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=300)
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.mode, args.max_users,
        args.top_products, args.novel_per_user, args.rounds)
