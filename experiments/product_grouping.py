"""Evaluate a Louvain product-group retriever against discovery baselines.

Communities are fit only from training users' completed baskets. Validation
targets are used only to measure candidate recall and the resulting top-five
list; this script does not fit or promote a recommendation model.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from experiments.basket_pilot import _targets
from experiments.discovery_retrieval import _measure
from grocery_recommender.data import load_bundle, prepare_data
from grocery_recommender.discovery_fusion import fuse_sources, rank_source
from grocery_recommender.modeling import split_user_ids
from grocery_recommender.sparse_discovery import (
    fit_popularity_index,
    fit_sparse_basket_index,
    retrieve_popular_index,
    retrieve_source,
)
from grocery_recommender.temporal_neighbors import fit_temporal_neighbors


def cluster_products(similarity: csr_matrix, product_ids: np.ndarray,
                     resolution: float, seed: int) -> tuple[np.ndarray, dict]:
    """Cluster the weighted co-basket graph and give isolates own groups."""
    symmetric = similarity.maximum(similarity.T).tocsr()
    symmetric.setdiag(0)
    symmetric.eliminate_zeros()
    graph = nx.from_scipy_sparse_array(symmetric, edge_attribute="weight")
    communities = nx.algorithms.community.louvain_communities(
        graph, weight="weight", resolution=resolution, seed=seed,
    )
    product_group = np.arange(len(product_ids), dtype=np.int32)
    graph_positions = np.asarray(sorted(graph.nodes), dtype=np.int64)
    group_count = 0
    sizes = []
    for community in communities:
        members = np.fromiter(community, dtype=np.int64)
        product_group[members] = group_count
        sizes.append(len(members))
        group_count += 1
    next_group = group_count
    for position in range(len(product_ids)):
        if position not in graph:
            product_group[position] = next_group
            next_group += 1
    degrees = np.diff(symmetric.indptr)
    nontrivial_sizes = [size for size in sizes if size > 1]
    stats = {
        "catalog_products": int(graph.number_of_nodes()),
        "products_with_graph_edges": int(np.count_nonzero(degrees)),
        "graph_edges_undirected": int(graph.number_of_edges()),
        "community_count": len(communities),
        "community_count_size_gt_1": len(nontrivial_sizes),
        "singleton_community_count": int(sum(size == 1 for size in sizes)),
        "community_size_median": float(np.median(sizes)) if sizes else 0.0,
        "community_size_p90": float(np.quantile(sizes, 0.9)) if sizes else 0.0,
        "largest_community_size": int(max(sizes, default=0)),
        "isolated_products": int(len(product_ids) - np.count_nonzero(degrees)),
        "resolution": resolution,
    }
    return product_group, stats


def group_candidates(anchors: pl.DataFrame, known: pl.DataFrame,
                     product_ids: np.ndarray, product_groups: np.ndarray,
                     popularity: dict[int, float], per_user: int) -> pl.DataFrame:
    """Use last-basket groups as seeds; rank group members by support/popularity."""
    group_by_product = dict(zip(product_ids.tolist(), product_groups.tolist()))
    known_by_user = dict(known.group_by("user_id").agg("product_id").iter_rows())
    anchor_by_user = dict(anchors.group_by("user_id").agg("product_id").iter_rows())
    rows = []
    for user_id, anchor_items in anchor_by_user.items():
        support = Counter(group_by_product[int(pid)] for pid in anchor_items
                          if int(pid) in group_by_product)
        if not support:
            continue
        owned = set(known_by_user.get(user_id, []))
        scores = []
        for product_id, group_id in group_by_product.items():
            if product_id in owned or group_id not in support:
                continue
            # A group receives more support when several last-basket products
            # belong to it; global frequency breaks ties within the group.
            score = support[group_id] * np.log1p(popularity.get(product_id, 0.0))
            if score > 0:
                scores.append((product_id, float(score), support[group_id]))
        scores.sort(key=lambda item: (-item[1], -item[2], item[0]))
        rows.extend((int(user_id), pid, score, count)
                    for pid, score, count in scores[:per_user])
    return pl.DataFrame(rows, schema={
        "user_id": pl.Int64, "product_id": pl.Int64,
        "group_score": pl.Float32, "anchor_group_support": pl.Int32,
    }, orient="row")


def run(data_dir: Path, output_dir: Path, max_users: int = 10000,
        budget: int = 80, resolution: float = 1.0, seed: int = 42) -> dict:
    prepared = prepare_data(data_dir)
    labeled = prepared.orders.filter(pl.col("eval_set") == "train")["user_id"].to_numpy()
    original_train = split_user_ids(labeled)[0]
    sampled = np.random.default_rng(2026).choice(
        original_train, size=min(max_users, len(original_train)), replace=False,
    )
    train_users, validation_users, _ = split_user_ids(sampled)
    bundle = load_bundle(data_dir, user_ids=sampled.tolist(), prepared=prepared)
    known_all = (bundle.history_lines.join(
        bundle.history_orders.select("order_id", "user_id"), on="order_id"
    ).select("user_id", "product_id").unique())
    anchors_all, targets_all, sizes_all = _targets(bundle, "pre_order")
    known = known_all.filter(pl.col("user_id").is_in(validation_users))
    anchors = anchors_all.filter(pl.col("user_id").is_in(validation_users))
    targets = targets_all.filter(pl.col("user_id").is_in(validation_users))
    sizes = sizes_all.filter(pl.col("user_id").is_in(validation_users))
    novel_targets = targets.join(known, on=["user_id", "product_id"], how="anti")
    repeats = targets.join(known, on=["user_id", "product_id"], how="semi")

    print("Fitting co-basket product graph on training customers...", flush=True)
    basket_index = fit_sparse_basket_index(bundle, train_users)
    product_groups, group_stats = cluster_products(
        basket_index.similarity, basket_index.product_ids, resolution, seed,
    )
    popularity_index = fit_popularity_index(bundle, train_users)
    popularity = dict(zip(popularity_index.product_ids.tolist(),
                          popularity_index.frequency.tolist()))
    grouped = group_candidates(anchors, known, basket_index.product_ids,
                               product_groups, popularity, budget)
    grouped_ranked = rank_source(grouped, "group", 0)
    grouped_measure = _measure(grouped, novel_targets, repeats, sizes)
    grouped_top5 = grouped_ranked.filter(pl.col("source_rank") <= 5)
    grouped_top5_hits = grouped_top5.join(
        novel_targets, on=["user_id", "product_id"], how="inner",
    ).height

    print("Building same-split retrieval baselines...", flush=True)
    neighbor_index = fit_temporal_neighbors(bundle, train_users)
    neighbor, _ = neighbor_index.score_users(bundle, validation_users, known, 40)
    basket = retrieve_source(basket_index, anchors, known, 40, "basket")
    popular_full = retrieve_popular_index(popularity_index, known, budget)
    sources = [
        rank_source(neighbor, "neighbor", 0),
        rank_source(basket, "basket", 1),
        rank_source(popular_full, "popularity", 2),
    ]
    behavioral_fill = fuse_sources(sources, budget)
    baseline_measure = _measure(behavioral_fill, novel_targets, repeats, sizes)
    popularity_budget = retrieve_popular_index(popularity_index, known, budget)
    popularity_measure = _measure(popularity_budget, novel_targets, repeats, sizes)

    report = {
        "method": "weighted co-basket graph Louvain communities",
        "max_users": int(len(sampled)),
        "train_users": int(len(train_users)),
        "validation_users": int(len(validation_users)),
        "validation_novel_target_items": int(novel_targets.height),
        "catalog_products": int(bundle.products.height),
        "candidate_budget": budget,
        "seed": seed,
        "cluster_graph": group_stats,
        "validation": {
            "group_retriever": grouped_measure,
            "group_top5_novel_hits": int(grouped_top5_hits),
            "group_top5_precision": grouped_top5_hits / (5 * len(validation_users)),
            "behavioral_neighbor_basket_popularity_fill": baseline_measure,
            "popularity_only": popularity_measure,
        },
        "interpretation": "Validation is used for model comparison; this experiment "
                          "does not promote or refit a production model.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    grouped.write_parquet(output_dir / "validation_group_candidates.parquet")
    joblib.dump({"product_ids": basket_index.product_ids,
                 "product_groups": product_groups,
                 "stats": group_stats}, output_dir / "product_groups.joblib")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/product_grouping_10k"))
    parser.add_argument("--max-users", type=int, default=10000)
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.max_users, args.budget,
        args.resolution, args.seed)


if __name__ == "__main__":
    main()
