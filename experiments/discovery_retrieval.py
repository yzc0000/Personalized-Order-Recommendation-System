"""Compare catalog-wide discovery sources before fitting a mixed ranker."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from experiments.basket_pilot import _targets
from grocery_recommender.data import load_bundle, prepare_data
from grocery_recommender.discovery_fusion import fuse_sources, rank_source as _rank_source
from grocery_recommender.modeling import split_user_ids
from grocery_recommender.sparse_discovery import (fit_content_index,
                                                   fit_popularity_index,
                                                   fit_sparse_basket_index,
                                                   retrieve_popular_index,
                                                   retrieve_source)
from grocery_recommender.temporal_neighbors import fit_temporal_neighbors


def _measure(candidates: pl.DataFrame, novel_targets: pl.DataFrame,
             repeats: pl.DataFrame, sizes: pl.DataFrame) -> dict:
    hits = candidates.join(novel_targets, on=["user_id", "product_id"]).height
    positives = pl.concat([repeats, candidates.join(novel_targets,
                    on=["user_id", "product_id"]).select(repeats.columns)])
    per_user = positives.group_by("user_id").agg(pl.len().alias("count"))
    oracle_hits = int(sizes.join(per_user, on="user_id", how="left")["count"]
                      .fill_null(0).clip(upper_bound=5).sum())
    return {"candidates": candidates.height, "novel_hits": hits,
            "novel_recall": hits / novel_targets.height,
            "candidate_oracle_p5": oracle_hits / (5 * sizes.height)}


def run(data_dir: Path, output_dir: Path, max_users: int = 10000,
        per_source: int = 40, budget: int = 80) -> dict:
    if max_users < 100 or per_source < 1 or budget < 1:
        raise ValueError("At least 100 users and positive candidate budgets are needed")
    prepared = prepare_data(data_dir)
    labeled = prepared.orders.filter(pl.col("eval_set") == "train")["user_id"].to_numpy()
    original_train = split_user_ids(labeled)[0]
    sampled = np.random.default_rng(2026).choice(
        original_train, size=min(max_users, len(original_train)), replace=False,
    )
    train_users, validation_users, test_users = split_user_ids(sampled)
    bundle = load_bundle(data_dir, user_ids=sampled.tolist(), prepared=prepared)
    print(f"Loaded {len(sampled)} users; fitting sparse basket graph...", flush=True)
    basket_index = fit_sparse_basket_index(bundle, train_users)
    print(f"Basket graph has {basket_index.similarity.nnz} directed links; "
          "fitting content and customer neighbors...", flush=True)
    content_index = fit_content_index(bundle)
    neighbor_index = fit_temporal_neighbors(bundle, train_users)
    popularity_index = fit_popularity_index(bundle, train_users)
    known_all = (bundle.history_lines.join(
        bundle.history_orders.select("order_id", "user_id"), on="order_id")
        .select("user_id", "product_id").unique())
    anchors_all, targets_all, sizes_all = _targets(bundle, "pre_order")

    def evaluate_split(name: str, users: np.ndarray):
        known = known_all.filter(pl.col("user_id").is_in(users))
        anchors = anchors_all.filter(pl.col("user_id").is_in(users))
        targets = targets_all.filter(pl.col("user_id").is_in(users))
        sizes = sizes_all.filter(pl.col("user_id").is_in(users))
        repeat = targets.join(known, on=["user_id", "product_id"], how="semi")
        novel_targets = targets.join(known, on=["user_id", "product_id"], how="anti")
        print(f"Retrieving {name} candidates from four sources...", flush=True)
        neighbor, _ = neighbor_index.score_users(bundle, users, known, per_source)
        popular_at_budget = retrieve_popular_index(popularity_index, known, budget)
        raw_sources = {
            "neighbor": neighbor,
            "basket": retrieve_source(basket_index, anchors, known, per_source, "basket"),
            "content": retrieve_source(content_index, anchors, known, per_source, "content"),
            "popularity": retrieve_popular_index(popularity_index, known, per_source),
        }
        ranked_sources = [_rank_source(raw_sources[source], source, index)
                          for index, source in enumerate(raw_sources)]
        summary = {
            "users": len(users), "target_items": targets.height,
            "novel_target_items": novel_targets.height,
            "repeat_candidate_recall": repeat.height / targets.height,
            "sources": {source: _measure(candidates, novel_targets, repeat, sizes)
                        for source, candidates in raw_sources.items()},
            "popularity_at_fusion_budget": _measure(
                popular_at_budget, novel_targets, repeat, sizes,
            ),
            "fusion": {}, "fusion_behavioral": {}, "fusion_behavioral_fill": {},
            "unique_source_hits_at_40": {},
        }
        ranked_popular_full = _rank_source(popular_at_budget, "popularity", 3)
        candidate_options = {}
        for cap in sorted(set((40, 80, 160, budget))):
            candidates = fuse_sources(ranked_sources, cap)
            assert candidates.join(known, on=["user_id", "product_id"]).is_empty()
            summary["fusion"][str(cap)] = _measure(
                candidates, novel_targets, repeat, sizes,
            )
            behavioral = fuse_sources(
                [ranked_sources[0], ranked_sources[1], ranked_sources[3]], cap,
            )
            summary["fusion_behavioral"][str(cap)] = _measure(
                behavioral, novel_targets, repeat, sizes,
            )
            behavioral_fill = fuse_sources(
                [ranked_sources[0], ranked_sources[1], ranked_popular_full], cap,
            )
            summary["fusion_behavioral_fill"][str(cap)] = _measure(
                behavioral_fill, novel_targets, repeat, sizes,
            )
            if cap == budget:
                candidate_options = {"all_four": candidates,
                                     "behavioral_three": behavioral,
                                     "behavioral_fill": behavioral_fill,
                                     "popularity_only": fuse_sources(
                                         [ranked_popular_full], cap)}
        for source, candidates in raw_sources.items():
            other = pl.concat([item.select("user_id", "product_id")
                               for name, item in raw_sources.items() if name != source])
            exclusive = (candidates.join(novel_targets, on=["user_id", "product_id"])
                         .join(other, on=["user_id", "product_id"], how="anti"))
            summary["unique_source_hits_at_40"][source] = exclusive.height
        return summary, candidate_options

    validation, _ = evaluate_split("validation", validation_users)
    comparison = {
        "popularity_only": validation["popularity_at_fusion_budget"]["novel_recall"],
        "all_four": validation["fusion"][str(budget)]["novel_recall"],
        "behavioral_three": validation["fusion_behavioral"][str(budget)]["novel_recall"],
        "behavioral_fill": validation["fusion_behavioral_fill"][str(budget)]["novel_recall"],
    }
    chosen_variant = max(comparison, key=comparison.get)
    test, options = evaluate_split("test", test_users)
    selected = options[chosen_variant]
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(basket_index, output_dir / "basket_index.joblib")
    joblib.dump(content_index, output_dir / "content_index.joblib")
    joblib.dump(neighbor_index, output_dir / "neighbor_index.joblib")
    joblib.dump(popularity_index, output_dir / "popularity_index.joblib")
    selected.write_parquet(output_dir / "test_novel_candidates.parquet")
    test_known = known_all.filter(pl.col("user_id").is_in(test_users))
    test_novel_targets = targets_all.filter(pl.col("user_id").is_in(test_users)).join(
        test_known, on=["user_id", "product_id"], how="anti",
    )
    preview_hits = selected.filter(pl.col("rank") <= 5).join(
        test_novel_targets, on=["user_id", "product_id"], how="inner",
    ).height
    report = {
        "decision_time": "before next order starts",
        "train_users": len(train_users), "validation": validation, "test": test,
        "catalog_products": bundle.products.height,
        "indexed_basket_products": int(np.count_nonzero(np.diff(
            basket_index.similarity.indptr))),
        "sparse_basket_edges": int(basket_index.similarity.nnz),
        "content_vocabulary_size": content_index.vocabulary_size,
        "per_source_limit": per_source, "fusion_limit": budget,
        "selected_fusion": chosen_variant,
        "validation_variant_novel_recall_at_budget": comparison,
        "test_new_only_fusion_p5": preview_hits / (5 * len(test_users)),
        "test_new_only_fusion_hits": preview_hits,
        "selected_by": f"validation novel recall at {budget} candidates/customer",
        "note": "The test split is inside the original full-run training population; "
                "candidate oracle sees the true items and is only a ranking ceiling.",
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({"validation": validation, "test": test}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/discovery_retrieval"))
    parser.add_argument("--max-users", type=int, default=10000)
    parser.add_argument("--per-source", type=int, default=40)
    parser.add_argument("--budget", type=int, default=80)
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.max_users, args.per_source, args.budget)
