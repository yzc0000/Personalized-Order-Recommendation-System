import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import polars as pl

from grocery_recommender.cli import run_recommend
from grocery_recommender.hybrid import combine_recommendations


def repeat_list(count=3):
    return pl.DataFrame({
        "user_id": [1] * 6,
        "product_id": list(range(1, 7)),
        "product_name": [f"Known {pid}" for pid in range(1, 7)],
        "rank": list(range(1, 7)),
        "score": [0.9] * count + [0.1] * (6 - count),
    })


def discovery_list():
    # A previously purchased item below the cutoff and a duplicate must both
    # be excluded before the two-slot cap is applied.
    return pl.DataFrame({
        "user_id": [1] * 5, "product_id": [6, 9, 9, 10, 11],
        "product_name": ["Known 6", "New 9", "New 9", "New 10", "New 11"],
        "preview_rank": [1, 2, 3, 4, 5], "score": [0.999] * 5,
    })


class HybridTests(unittest.TestCase):
    def test_cap_and_available_slots_for_every_repeat_count(self):
        for count in range(7):
            with self.subTest(accepted_repeats=count):
                result = combine_recommendations(repeat_list(count), discovery_list())
                retained_count = min(count, 5)
                expected_new = min(2, 5 - retained_count)
                self.assertEqual(result.height, retained_count + expected_new)
                self.assertEqual(result["product_id"].to_list(),
                                 list(range(1, retained_count + 1)) + [9, 10][:expected_new])
                self.assertEqual(result["rank"].to_list(), list(range(1, result.height + 1)))
                self.assertEqual(result["recommendation_type"].to_list(),
                                 ["Repeat"] * retained_count + ["Discovery"] * expected_new)

    def test_cutoff_boundary_and_discovery_probability_labels(self):
        repeats = repeat_list(1).with_columns(
            pl.Series("score", [0.5, 0.499, 0.1, 0.1, 0.1, 0.1]))
        result = combine_recommendations(repeats, discovery_list())
        self.assertEqual(result["product_id"].to_list(), [1, 9, 10])
        self.assertEqual(result["estimated_reorder_probability"].to_list(),
                         [0.5, None, None])

    def test_no_discovery_does_not_force_five_items(self):
        empty = discovery_list().head(0)
        result = combine_recommendations(repeat_list(3), empty)
        self.assertEqual(result.height, 3)
        disabled = combine_recommendations(repeat_list(0), discovery_list(), max_discovery=0)
        self.assertEqual(disabled.height, 0)

    def test_limits_are_applied_per_customer(self):
        repeats = pl.concat([repeat_list(0), repeat_list(3).with_columns(pl.lit(2).alias("user_id"))],
                            how="vertical_relaxed")
        discoveries = pl.concat([discovery_list(), discovery_list().with_columns(
            pl.lit(2).alias("user_id"))], how="vertical_relaxed")
        result = combine_recommendations(repeats, discoveries)
        self.assertEqual(result.filter(pl.col("user_id") == 1).height, 2)
        self.assertEqual(result.filter(pl.col("user_id") == 2).height, 5)

    def test_invalid_limits_fail(self):
        for kwargs in ({"k": 0}, {"max_discovery": 3}, {"min_probability": 1.1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                combine_recommendations(repeat_list(), discovery_list(), **kwargs)

    def run_cli(self, count=3, max_discovery=2, ranking="specialist_model"):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "model.joblib").touch()
            frame = repeat_list(count).select("user_id", "product_id", "product_name")
            scorer = SimpleNamespace(name="xgboost", feature_columns=(),
                                     predict=lambda _: repeat_list(count)["score"].to_numpy())
            args = SimpleNamespace(output_dir=model_dir, data_dir=Path("."),
                                   user_id=1, k=5, min_probability=0.5,
                                   max_discovery=max_discovery,
                                   discovery_dir=Path("indices"),
                                   discovery_ranker_dir=Path("specialist"))
            bundle = object()
            preview = discovery_list().filter(pl.col("product_id").is_in([9, 10])).unique(
                subset=["user_id", "product_id"], maintain_order=True)
            with patch("grocery_recommender.cli.joblib.load", return_value=scorer), \
                    patch("grocery_recommender.cli.load_bundle", return_value=bundle), \
                    patch("grocery_recommender.cli.validate_bundle"), \
                    patch("grocery_recommender.cli.build_features",
                          return_value=SimpleNamespace(candidates=frame)), \
                    patch("grocery_recommender.cli.preview_candidates",
                          return_value=(preview, {"ranking": ranking})) as retrieve, \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                run_recommend(args)
            return output.getvalue(), retrieve, bundle, args

    def test_cli_combines_models_and_labels_new_items(self):
        output, retrieve, bundle, args = self.run_cli()
        retrieve.assert_called_once_with(bundle, args.discovery_dir,
                                         args.discovery_ranker_dir, 2)
        self.assertIn("Discovery", output)
        self.assertIn("Repeat", output)
        self.assertNotIn("0.999", output)

    def test_cli_skips_discovery_when_full_or_disabled(self):
        for count, cap in ((5, 2), (3, 0)):
            with self.subTest(count=count, cap=cap):
                _, retrieve, _, _ = self.run_cli(count=count, max_discovery=cap)
                retrieve.assert_not_called()

    def test_cli_requires_the_discovery_model(self):
        with self.assertRaisesRegex(ValueError, "compatible discovery specialist"):
            self.run_cli(ranking="source_fusion")


if __name__ == "__main__":
    unittest.main()
