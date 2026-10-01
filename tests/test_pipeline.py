import unittest

import numpy as np
import polars as pl

from grocery_recommender.data import DataBundle, validate_bundle
from grocery_recommender.features import (FEATURE_COLUMNS, SEQUENCE_PILOT_COLUMNS,
                                          REPLENISHMENT_PILOT_COLUMNS,
                                          build_features)
from grocery_recommender.metrics import ranking_metrics
from grocery_recommender.modeling import Scorer, fit_scorer, split_user_ids
from grocery_recommender.augmentation import historical_training_examples
from grocery_recommender.decisions import choose_threshold, decision_metrics, probability_metrics
from grocery_recommender.novel import item_collaborative_candidates
from grocery_recommender.co_basket import (fit_co_basket_index,
                                           retrieve_co_basket_candidates)
from grocery_recommender.mixed import build_mixed_candidates
from experiments.novel_ranker_pilot import _candidates
from experiments.basket_pilot import _targets
from grocery_recommender.population_features import fit_product_priors
from grocery_recommender.temporal_neighbors import fit_temporal_neighbors
from grocery_recommender.sparse_discovery import (fit_popularity_index,
                                                   fit_sparse_basket_index,
                                                   retrieve_popular_index)
from grocery_recommender.discovery_fusion import fuse_sources, rank_source as _rank_source
from grocery_recommender.discovery_features import (MIXED_COLUMNS,
                                                      build_discovery_candidates)


def small_bundle() -> DataBundle:
    history_orders = pl.DataFrame({
        "order_id": [10, 11], "user_id": [1, 1], "eval_set": ["prior", "prior"],
        "order_number": [1, 2], "order_dow": [1, 2], "order_hour_of_day": [8, 8],
        "days_since_prior_order": [None, 7.0],
    })
    history_lines = pl.DataFrame({
        "order_id": [10, 10, 11], "product_id": [1, 2, 1],
        "add_to_cart_order": [1, 2, 1], "reordered": [0, 0, 1],
    })
    target_orders = pl.DataFrame({
        "order_id": [12], "user_id": [1], "eval_set": ["train"],
        "order_number": [3], "order_dow": [3], "order_hour_of_day": [9],
        "days_since_prior_order": [4.0],
    })
    target_lines = pl.DataFrame({
        "order_id": [12, 12], "product_id": [1, 3],
        "add_to_cart_order": [1, 2], "reordered": [1, 0],
    })
    products = pl.DataFrame({
        "product_id": [1, 2, 3], "product_name": ["Milk", "Eggs", "Bread"],
        "aisle_id": [1, 1, 2], "department_id": [1, 1, 1],
    })
    product_stats = pl.DataFrame({
        "product_id": [1, 2, 3], "product_purchase_count": [2, 1, 1],
        "product_reorder_rate": [0.5, 0.0, 0.0],
    })
    return DataBundle(history_orders, history_lines, target_orders, target_lines,
                      products, product_stats)


class PipelineTests(unittest.TestCase):
    def test_features_use_history_and_new_items_set_candidate_ceiling(self):
        bundle = small_bundle()
        validate_bundle(bundle)
        features = build_features(bundle, include_sequence_pilot=True)
        self.assertEqual(features.candidates.height, 2)
        self.assertEqual(features.candidate_hits, 1)
        self.assertEqual(features.target_item_count, 2)
        by_product = {row["product_id"]: row for row in features.candidates.to_dicts()}
        self.assertEqual(by_product[1]["up_purchase_count"], 2)
        self.assertEqual(by_product[2]["up_order_gap"], 1)
        self.assertEqual(by_product[2]["up_days_since_last_purchase"], 7)
        self.assertNotIn("up_days_at_next_order", features.candidates.columns)
        self.assertEqual(by_product[1]["up_recent3_count"], 2)
        self.assertEqual(by_product[1]["up_last2_count"], 2)
        self.assertEqual(by_product[1]["up_previous_order_gap"], 1)
        self.assertEqual(by_product[1]["label"], 1)
        self.assertEqual(by_product[2]["label"], 0)

        changed = small_bundle()
        changed.target_orders = changed.target_orders.with_columns(
            pl.lit(6).alias("order_dow"), pl.lit(22).alias("order_hour_of_day"),
            pl.lit(30.0).alias("days_since_prior_order"),
        )
        changed.target_lines = pl.DataFrame({
            "order_id": [12, 12], "product_id": [2, 3],
            "add_to_cart_order": [1, 2], "reordered": [1, 0],
        })
        changed_features = build_features(changed, include_sequence_pilot=True)
        original = features.candidates.sort("product_id").select(FEATURE_COLUMNS)
        new = changed_features.candidates.sort("product_id").select(FEATURE_COLUMNS)
        self.assertTrue(original.equals(new))
        original_sequence = features.candidates.sort("product_id").select(
            SEQUENCE_PILOT_COLUMNS)
        changed_sequence = changed_features.candidates.sort("product_id").select(
            SEQUENCE_PILOT_COLUMNS)
        self.assertTrue(original_sequence.equals(changed_sequence))
        self.assertNotIn("up_recent5_trend", build_features(small_bundle()).candidates.columns)

    def test_ranking_metrics_include_unseen_target_products(self):
        features = build_features(small_bundle())
        candidates = features.candidates.sort("product_id")
        scores = np.array([0.9, 0.1])
        result = ranking_metrics(candidates, features.target_sizes, scores, k=2)
        self.assertAlmostEqual(result["precision_at_2"], 0.5)
        self.assertAlmostEqual(result["recall_at_2"], 0.5)
        expected_ndcg = 1 / (1 + 1 / np.log2(3))
        self.assertAlmostEqual(result["ndcg_at_2"], expected_ndcg)

    def test_user_split_is_independent_of_input_order(self):
        users = np.arange(1, 101)
        first = split_user_ids(users)
        second = split_user_ids(users[::-1])
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(len(set(first[0]) & set(first[1])), 0)

    def test_historical_snapshot_sees_only_earlier_orders(self):
        bundle = small_bundle()
        bundle.history_orders = pl.concat([bundle.history_orders, bundle.target_orders])
        bundle.history_lines = pl.concat([bundle.history_lines, bundle.target_lines])
        bundle.target_orders = pl.DataFrame({
            "order_id": [13], "user_id": [1], "eval_set": ["train"],
            "order_number": [4], "order_dow": [4], "order_hour_of_day": [10],
            "days_since_prior_order": [5.0],
        })
        bundle.target_lines = pl.DataFrame({
            "order_id": [13], "product_id": [3],
            "add_to_cart_order": [1], "reordered": [1],
        })
        pseudo = historical_training_examples(bundle, np.array([1]), snapshots=1)
        self.assertEqual(set(pseudo["product_id"].to_list()), {1, 2})
        labels = dict(zip(pseudo["product_id"].to_list(), pseudo["label"].to_list()))
        self.assertEqual(labels, {1: 1, 2: 0})
        self.assertEqual(pseudo["user_order_count"].unique().to_list(), [2])
        self.assertEqual(pseudo["snapshot_id"].unique().to_list(), [3])
        self.assertNotIn("next_days_since_prior_order", pseudo.columns)

    def test_ranker_keeps_historical_baskets_as_separate_queries(self):
        class CapturingRanker:
            def fit(self, matrix, labels, qid):
                self.queries = qid.copy()
                self.labels = labels.copy()

        rows = pl.DataFrame({
            "user_id": [2, 1, 1, 1, 2, 1],
            "snapshot_id": [3, 4, 3, 4, 3, 3],
            "up_purchase_share": [.1, .8, .2, .7, .9, .3],
            "label": [0, 1, 1, 0, 1, 0],
        })
        ranker = CapturingRanker()
        fit_scorer(Scorer("xgboost_ranker", ranker,
                          ("up_purchase_share",)), rows)
        self.assertEqual(ranker.queries.tolist(), [1003, 1003, 1004,
                                                    1004, 2003, 2003])
        self.assertEqual(sorted(set(ranker.queries)), [1003, 1004, 2003])

    def test_variable_length_reorder_decision_can_return_zero_items(self):
        labels = np.array([1, 0, 1, 0])
        scores = np.array([0.9, 0.2, 0.4, 0.1])
        users = np.array([1, 1, 2, 2])
        metrics = decision_metrics(labels, scores, users, threshold=0.5)
        self.assertEqual(metrics["predicted_reorders"], 1)
        self.assertEqual(metrics["users_with_no_predictions"], 1)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall_of_reorders"], 0.5)

    def test_cutoff_selection_uses_f0_5_on_supplied_validation_rows(self):
        labels = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.4, 0.3, 0.1])
        users = np.array([1, 2, 1, 2])
        threshold, trials = choose_threshold(labels, scores, users)
        self.assertGreater(threshold, 0.3)
        self.assertLessEqual(threshold, 0.4)
        self.assertEqual(max(row["f0_5"] for row in trials), 1.0)
        quality = probability_metrics(labels, scores)
        self.assertEqual(sum(row["count"] for row in quality["calibration_bins"]), 4)

    def test_collaborative_candidates_are_new_to_each_customer(self):
        known = pl.DataFrame({
            "user_id": [1, 1, 2, 2, 3, 3],
            "product_id": [1, 2, 1, 3, 1, 3],
        })
        stats = pl.DataFrame({
            "product_id": [1, 2, 3, 4],
            "product_purchase_count": [3, 1, 2, 1],
        })
        novel, known_scores = item_collaborative_candidates(
            known, stats, np.array([1, 2, 3]), top_products=4, per_user=1
        )
        self.assertEqual(novel.join(known, on=["user_id", "product_id"]).height, 0)
        self.assertLessEqual(novel.group_by("user_id").len()["len"].max(), 1)
        self.assertEqual(known_scores.height, known.height)

    def test_new_candidate_can_be_a_positive_next_basket_label(self):
        bundle = small_bundle()
        base = build_features(bundle)
        novel = pl.DataFrame({
            "user_id": [1], "product_id": [3], "collaborative_score": [0.7],
        })
        known_scores = pl.DataFrame({
            "user_id": [1, 1], "product_id": [1, 2],
            "collaborative_score": [0.2, 0.1],
        })
        _, mixed = _candidates(bundle, base, novel, known_scores)
        self.assertEqual(mixed.height, 3)
        self.assertEqual(mixed.filter(pl.col("is_novel") == 1)["label"].to_list(), [1])

    def test_co_basket_index_uses_only_prior_orders(self):
        bundle = small_bundle()
        first = fit_co_basket_index(bundle, np.array([1]), top_products=2)
        changed = small_bundle()
        changed.target_lines = pl.DataFrame({
            "order_id": [12, 12], "product_id": [2, 3],
            "add_to_cart_order": [1, 2], "reordered": [1, 0],
        })
        second = fit_co_basket_index(changed, np.array([1]), top_products=2)
        np.testing.assert_array_equal(first.product_ids, second.product_ids)
        np.testing.assert_array_equal(first.similarity, second.similarity)
        anchor = pl.DataFrame({"user_id": [1], "product_id": [1]})
        known = pl.DataFrame({"user_id": [1], "product_id": [1]})
        novel, _ = retrieve_co_basket_candidates(first, anchor, known, per_user=1)
        self.assertEqual(novel["product_id"].to_list(), [2])

    def test_session_labels_exclude_observed_cart_and_keep_new_item(self):
        bundle = small_bundle()
        base = build_features(bundle)
        anchor = pl.DataFrame({"user_id": [1], "product_id": [1]})
        novel = pl.DataFrame({
            "user_id": [1], "product_id": [3], "basket_affinity": [0.4],
        })
        known_scores = pl.DataFrame({
            "user_id": [1, 1], "product_id": [1, 2],
            "basket_affinity": [0.2, 0.1],
        })
        remaining = pl.DataFrame({"user_id": [1], "product_id": [3]})
        repeat, mixed = build_mixed_candidates(
            bundle, base, novel, known_scores, anchor, remaining,
            exclude_anchors=True,
        )
        self.assertEqual(repeat["product_id"].to_list(), [2])
        self.assertEqual(set(mixed["product_id"].to_list()), {2, 3})
        self.assertEqual(mixed.filter(pl.col("product_id") == 3)["label"].item(), 1)

    def test_two_item_cart_can_have_no_remaining_products(self):
        anchors, remaining, sizes = _targets(small_bundle(), "session")
        self.assertEqual(set(anchors["product_id"].to_list()), {1, 3})
        self.assertTrue(remaining.is_empty())
        self.assertEqual(sizes["target_size"].to_list(), [0])

    def test_replenishment_features_and_product_priors_ignore_hidden_basket(self):
        original = small_bundle()
        frame = build_features(original, include_replenishment_pilot=True).candidates
        by_product = {row["product_id"]: row for row in frame.to_dicts()}
        self.assertEqual(by_product[1]["up_trailing_streak"], 2)
        self.assertEqual(by_product[2]["up_trailing_streak"], 0)
        self.assertEqual(by_product[1]["user_aisle_diversity"], 2)
        self.assertAlmostEqual(by_product[1]["up_aisle_purchase_fraction"], 2 / 3)
        priors = fit_product_priors(original, np.array([1])).sort("product_id")
        self.assertAlmostEqual(priors.filter(pl.col("product_id") == 1)
                               ["product_next_order_rate"].item(), 2 / 11)
        changed = small_bundle()
        changed.target_lines = pl.DataFrame({
            "order_id": [12, 12], "product_id": [2, 3],
            "add_to_cart_order": [1, 2], "reordered": [1, 0],
        })
        feature_names = REPLENISHMENT_PILOT_COLUMNS
        self.assertTrue(frame.sort("product_id").select(feature_names).equals(
            build_features(changed, include_replenishment_pilot=True)
            .candidates.sort("product_id").select(feature_names)))
        self.assertTrue(priors.equals(fit_product_priors(changed, np.array([1]))
                                     .sort("product_id")))

    def test_customer_neighbors_retrieve_unseen_products_without_target_data(self):
        bundle = small_bundle()
        second_orders = pl.DataFrame({
            "order_id": [20, 21], "user_id": [2, 2], "eval_set": ["prior", "prior"],
            "order_number": [1, 2], "order_dow": [1, 2],
            "order_hour_of_day": [8, 9], "days_since_prior_order": [None, 7.0],
        })
        second_lines = pl.DataFrame({
            "order_id": [20, 20, 21, 21], "product_id": [1, 3, 1, 3],
            "add_to_cart_order": [1, 2, 1, 2], "reordered": [0, 0, 1, 1],
        })
        bundle.history_orders = pl.concat([bundle.history_orders, second_orders])
        bundle.history_lines = pl.concat([bundle.history_lines, second_lines])
        index = fit_temporal_neighbors(bundle, np.array([2]), neighbors=1)
        known = pl.DataFrame({"user_id": [1, 1], "product_id": [1, 2]})
        novel, own = index.score_users(bundle, np.array([1]), known, per_user=2)
        self.assertEqual(novel["product_id"].to_list(), [3])
        self.assertEqual(set(own["product_id"].to_list()), {1, 2})
        changed = small_bundle()
        changed.history_orders = bundle.history_orders
        changed.history_lines = bundle.history_lines
        changed.target_lines = changed.target_lines.with_columns(
            pl.lit(2).alias("product_id")
        )
        again, _ = index.score_users(changed, np.array([1]), known, per_user=2)
        self.assertTrue(novel.equals(again))

    def test_sparse_basket_graph_keeps_catalog_without_target_leakage(self):
        bundle = small_bundle()
        original = fit_sparse_basket_index(bundle, np.array([1]),
                                           min_pair_count=1, min_product_count=1)
        self.assertEqual(len(original.product_ids), 3)
        changed = small_bundle()
        changed.target_lines = changed.target_lines.with_columns(
            pl.lit(2).alias("product_id")
        )
        second = fit_sparse_basket_index(changed, np.array([1]),
                                         min_pair_count=1, min_product_count=1)
        np.testing.assert_array_equal(original.product_ids, second.product_ids)
        np.testing.assert_array_equal(original.similarity.toarray(),
                                      second.similarity.toarray())

    def test_saved_popularity_uses_only_fit_histories(self):
        bundle = small_bundle()
        original = fit_popularity_index(bundle, np.array([1]))
        known = pl.DataFrame({"user_id": [1], "product_id": [1]})
        result = retrieve_popular_index(original, known, per_user=2)
        self.assertEqual(result["product_id"].to_list(), [2])
        changed = small_bundle()
        changed.target_lines = changed.target_lines.with_columns(
            pl.lit(2).alias("product_id")
        )
        again = fit_popularity_index(changed, np.array([1]))
        np.testing.assert_array_equal(original.product_ids, again.product_ids)
        np.testing.assert_array_equal(original.frequency, again.frequency)

    def test_discovery_rank_features_ignore_hidden_next_basket(self):
        def frame(bundle):
            base = build_features(bundle)
            fused = pl.DataFrame({"user_id": [1], "product_id": [3], "rank": [1]})
            sources = {
                "neighbor": pl.DataFrame({"user_id": [1], "product_id": [3],
                                          "neighbor_score": [0.7]}),
                "basket": pl.DataFrame({"user_id": [1], "product_id": [3],
                                        "basket_score": [0.3]}),
                "popularity": pl.DataFrame({"user_id": [1], "product_id": [3],
                                            "popularity_score": [10.0]}),
            }
            neighbor_known = pl.DataFrame({
                "user_id": [1, 1], "product_id": [1, 2],
                "neighbor_score": [0.5, 0.1],
            })
            anchors = pl.DataFrame({"user_id": [1], "product_id": [1]})
            targets = bundle.target_lines.join(
                bundle.target_orders.select("order_id", "user_id"), on="order_id",
            ).select("user_id", "product_id")
            return build_discovery_candidates(
                bundle, base, fused, sources, neighbor_known, anchors, targets,
            )[1].sort("product_id")

        original = frame(small_bundle())
        changed = small_bundle()
        changed.target_lines = pl.DataFrame({
            "order_id": [12, 12], "product_id": [2, 3],
            "add_to_cart_order": [1, 2], "reordered": [1, 0],
        })
        second = frame(changed)
        self.assertTrue(original.select(MIXED_COLUMNS).equals(
            second.select(MIXED_COLUMNS)))
        self.assertNotEqual(original["label"].to_list(), second["label"].to_list())

    def test_discovery_fusion_deduplicates_and_respects_budget(self):
        a = pl.DataFrame({"user_id": [1, 1], "product_id": [3, 4],
                          "neighbor_score": [0.8, 0.4]})
        b = pl.DataFrame({"user_id": [1, 1], "product_id": [3, 5],
                          "basket_score": [0.9, 0.5]})
        fused = fuse_sources([_rank_source(a, "neighbor", 0),
                              _rank_source(b, "basket", 1)], budget=2)
        self.assertEqual(fused["product_id"].to_list(), [3, 4])
        self.assertEqual(fused["product_id"].n_unique(), fused.height)


if __name__ == "__main__":
    unittest.main()
