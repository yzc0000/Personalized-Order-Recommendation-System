"""Reproduce research checks without training or changing recommendation models."""

import hashlib
import json
from itertools import islice
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from experiments.basket_pilot import _targets
from grocery_recommender.co_basket import retrieve_co_basket_candidates
from grocery_recommender.data import load_bundle, prepare_data
from grocery_recommender.modeling import split_user_ids


def run(root: Path) -> dict:
    prepared = prepare_data(root)
    labeled = prepared.orders.filter(pl.col("eval_set") == "train")["user_id"].to_numpy()
    original_train, _, original_test = split_user_ids(labeled)
    pilot = json.loads((root / "artifacts/preorder_pilot/metrics.json").read_text())
    sampled = np.random.default_rng(42).choice(
        original_train, size=pilot["sample_users"], replace=False,
    )
    train_users, validation_users, _ = split_user_ids(sampled)
    bundle = load_bundle(root, user_ids=validation_users.tolist(), prepared=prepared)
    known = bundle.history_lines.join(
        bundle.history_orders.select("order_id", "user_id"), on="order_id",
    ).select("user_id", "product_id").unique()
    anchors, target, sizes = _targets(bundle, "pre_order")
    repeats = target.join(known, on=["user_id", "product_id"], how="semi")
    novel_targets = target.join(known, on=["user_id", "product_id"], how="anti")
    assert repeats.height == bundle.target_lines["reordered"].sum()
    index = joblib.load(root / "artifacts/preorder_pilot/co_basket_index.joblib")
    novel, _ = retrieve_co_basket_candidates(
        index, anchors, known, per_user=pilot["novel_per_user"],
    )
    assert novel.join(known, on=["user_id", "product_id"], how="inner").is_empty()
    reached = novel.join(novel_targets, on=["user_id", "product_id"], how="inner")
    in_catalog = novel_targets.filter(pl.col("product_id").is_in(index.product_ids))

    def oracle(positives: pl.DataFrame) -> float:
        counts = positives.group_by("user_id").len()
        return float(sizes.join(counts, on="user_id", how="left")["len"]
                     .fill_null(0).clip(upper_bound=5).sum() / (5 * sizes.height))

    # Compare against popularity from the same training histories, as well as
    # the existing all-prior global statistics. Neither baseline uses labels.
    training_orders = prepared.orders.filter(
        (pl.col("eval_set") == "prior") & pl.col("user_id").is_in(train_users),
    ).select("order_id")
    training_popularity = (
        pl.scan_csv(root / "order_products__prior.csv")
        .join(training_orders.lazy(), on="order_id", how="semi")
        .group_by("product_id").agg(pl.len().alias("product_purchase_count"))
        .collect(engine="streaming")
    )
    known_map = dict(known.group_by("user_id").agg("product_id").iter_rows())

    def popularity_result(statistics: pl.DataFrame, source: str) -> dict:
        popularity = statistics.sort(
            ["product_purchase_count", "product_id"], descending=[True, False],
        )["product_id"].to_list()
        popular_rows = []
        for uid in validation_users:
            owned = set(known_map[int(uid)])
            candidates = islice((pid for pid in popularity if pid not in owned),
                                pilot["novel_per_user"])
            popular_rows.extend((int(uid), pid, rank + 1)
                                for rank, pid in enumerate(candidates))
        popular = pl.DataFrame(popular_rows, orient="row",
                               schema=["user_id", "product_id", "rank"])
        hits = popular.join(novel_targets, on=["user_id", "product_id"])
        return {
            "statistics_source": source, "candidates": popular.height,
            "hits_at_40": hits.height,
            "novel_recall_at_40": hits.height / novel_targets.height,
            "precision_at_5": hits.filter(pl.col("rank") <= 5).height / (5 * len(validation_users)),
        }

    popular_training = popularity_result(training_popularity, "same 14k training-user prior histories as co-basket index")
    popular_all = popularity_result(prepared.product_stats, "all prior histories; different fitting population from co-basket index")
    anchor_coverage = anchors.with_columns(
        pl.col("product_id").is_in(index.product_ids).alias("indexed"),
    ).group_by("user_id").agg(pl.col("indexed").any())

    # Recompute the existing full-model result from saved recommendations and
    # raw target rows, independently of the saved label and metric columns.
    recs = pl.read_csv(root / "artifacts/full_v2/test_recommendations.csv")
    next_orders = prepared.orders.filter(pl.col("eval_set") == "train")
    truth = pl.read_csv(root / "order_products__train.csv").join(
        next_orders.select("order_id", "user_id"), on="order_id",
    ).select("user_id", "product_id")
    assert set(recs["user_id"].unique().to_list()) == set(original_test.tolist())
    assert recs.select(pl.struct("user_id", "product_id").n_unique()).item() == recs.height
    marked = recs.join(truth.with_columns(pl.lit(1).alias("raw_label")),
                       on=["user_id", "product_id"], how="left").with_columns(
        pl.col("raw_label").fill_null(0),
    )
    assert marked.filter(pl.col("label") != pl.col("raw_label")).is_empty()
    hits = int(marked.filter(pl.col("rank") <= 5)["raw_label"].sum())
    full_metrics = json.loads((root / "artifacts/full_v2/metrics.json").read_text())
    recomputed_p5 = hits / (5 * len(original_test))
    assert abs(recomputed_p5 - full_metrics["test_selected_model"]["5"]["precision_at_5"]) < 1e-12

    report = {
        "purpose": "Read-only metric and retrieval audit; no model selection or fitting",
        "source_script": "experiments/research_audit.py",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "full_model_recomputation": {
            "users": len(original_test), "top5_hits_from_raw_targets": hits,
            "precision_at_5": recomputed_p5,
            "saved_labels_agree_with_raw_target_membership": True,
        },
        "preorder_validation_retrieval": {
            "users": len(validation_users), "retriever_training_users": len(train_users),
            "population": "3,000 validation users from the existing 20k original-training-user pilot",
            "catalog_products": prepared.products.height,
            "indexed_products": len(index.product_ids),
            "indexed_catalog_fraction": len(index.product_ids) / prepared.products.height,
            "target_items": target.height, "repeat_target_items": repeats.height,
            "novel_target_items": novel_targets.height,
            "novel_target_items_in_index": in_catalog.height,
            "indexed_fraction_of_novel_targets": in_catalog.height / novel_targets.height,
            "novel_candidates": novel.height, "novel_candidate_hits": reached.height,
            "novel_retrieval_recall_at_40": reached.height / novel_targets.height,
            "recall_of_indexed_novel_targets": reached.height / in_catalog.height,
            "novel_candidate_positive_rate": reached.height / novel.height,
            "repeat_candidate_recall": repeats.height / target.height,
            "mixed_candidate_recall": (repeats.height + reached.height) / target.height,
            "repeat_oracle_precision_at_5": oracle(repeats),
            "mixed_oracle_precision_at_5": oracle(pl.concat([repeats, reached.select(repeats.columns)])),
            "all_catalog_oracle_precision_at_5": oracle(target),
            "users_without_indexed_anchor": anchor_coverage.filter(~pl.col("indexed")).height,
            "popular_unseen_same_history_baseline": popular_training,
            "popular_unseen_all_prior_baseline": popular_all,
        },
    }
    output = root / "reports/research_audit.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(Path(".")), indent=2))
