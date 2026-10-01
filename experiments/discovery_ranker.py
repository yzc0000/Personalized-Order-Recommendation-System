"""Test whether stronger novel retrieval improves the original five-slot panel."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from experiments.basket_pilot import _targets
from experiments.discovery_retrieval import _rank_source, fuse_sources
from grocery_recommender.data import load_bundle, prepare_data
from grocery_recommender.discovery_features import (MIXED_COLUMNS,
                                                      build_discovery_candidates)
from grocery_recommender.features import FEATURE_COLUMNS, build_features
from grocery_recommender.metrics import rank_candidates, ranking_metrics
from grocery_recommender.modeling import fit_scorer, new_scorer, split_user_ids
from grocery_recommender.sparse_discovery import retrieve_popular_index, retrieve_source


def _hits(frame: pl.DataFrame, scores: np.ndarray, users: np.ndarray) -> np.ndarray:
    ranked = rank_candidates(frame.select("user_id", "product_id", "label"), scores)
    counts = ranked.filter(pl.col("rank") <= 5).group_by("user_id").agg(
        pl.col("label").sum().alias("hits")
    )
    return (pl.DataFrame({"user_id": users}).join(counts, on="user_id", how="left",
                                             maintain_order="left")["hits"]
            .fill_null(0).to_numpy())


def _interval(difference: np.ndarray) -> list[float]:
    rng = np.random.default_rng(42)
    picks = rng.integers(0, len(difference), size=(1200, len(difference)))
    return [float(x) for x in np.quantile(difference[picks].mean(axis=1) / 5,
                                          [0.025, 0.975])]


def run(data_dir: Path, retrieval_dir: Path, output_dir: Path,
        max_users: int = 10000, rounds: int = 350) -> dict:
    retrieval = json.loads((retrieval_dir / "metrics.json").read_text())
    variant = retrieval["selected_fusion"]
    per_source = retrieval["per_source_limit"]
    budget = retrieval["fusion_limit"]
    prepared = prepare_data(data_dir)
    eligible = prepared.orders.filter(pl.col("eval_set") == "train")["user_id"].to_numpy()
    original_train = split_user_ids(eligible)[0]
    sampled = np.random.default_rng(2026).choice(
        original_train, size=min(max_users, len(original_train)), replace=False,
    )
    train_users, validation_users, test_users = split_user_ids(sampled)
    if len(train_users) != retrieval["train_users"]:
        raise ValueError("Retriever and ranker must use the identical training users")
    bundle = load_bundle(data_dir, user_ids=sampled.tolist(), prepared=prepared)
    base = build_features(bundle)
    anchors, targets, sizes = _targets(bundle, "pre_order")
    known = base.candidates.select("user_id", "product_id")
    basket_index = joblib.load(retrieval_dir / "basket_index.joblib")
    neighbor_index = joblib.load(retrieval_dir / "neighbor_index.joblib")
    popularity_index = joblib.load(retrieval_dir / "popularity_index.joblib")
    if not np.array_equal(neighbor_index.train_users, np.sort(train_users)):
        raise ValueError("Retriever training-user IDs differ from ranker split")
    print(f"Retrieving for {len(sampled)} users with {variant}...", flush=True)
    neighbor_novel, neighbor_known = neighbor_index.score_users(
        bundle, np.sort(sampled), known, per_user=per_source
    )
    sources = {
        "neighbor": neighbor_novel,
        "basket": retrieve_source(basket_index, anchors, known, per_source, "basket"),
        "popularity": retrieve_popular_index(
            popularity_index, known,
            budget if variant in {"behavioral_fill", "popularity_only"} else per_source,
        ),
    }
    if variant == "all_four":
        content_index = joblib.load(retrieval_dir / "content_index.joblib")
        sources["content"] = retrieve_source(
            content_index, anchors, known, per_source, "content",
        )
    sequence = (["popularity"] if variant == "popularity_only" else
                ["neighbor", "basket", "content", "popularity"])
    ranked = [_rank_source(sources[name], name, sequence.index(name))
              for name in sequence if name in sources]
    fused = fuse_sources(ranked, budget)
    assert fused.join(known, on=["user_id", "product_id"]).is_empty()

    repeat, mixed = build_discovery_candidates(
        bundle, base, fused, sources, neighbor_known, anchors, targets,
    )

    def fit(frame: pl.DataFrame, users: np.ndarray, columns: tuple[str, ...]):
        model = new_scorer("xgboost")
        model.feature_columns = columns
        model.estimator.set_params(n_estimators=rounds)
        return fit_scorer(model, frame.filter(pl.col("user_id").is_in(users)))

    print(f"Fitting repeat baseline on {len(train_users)} users...", flush=True)
    baseline = fit(repeat, train_users, tuple(FEATURE_COLUMNS))
    print("Fitting repeat/new ranker...", flush=True)
    challenger = fit(mixed, train_users, MIXED_COLUMNS)
    val_repeat = repeat.filter(pl.col("user_id").is_in(validation_users))
    val_mixed = mixed.filter(pl.col("user_id").is_in(validation_users))
    val_sizes = sizes.filter(pl.col("user_id").is_in(validation_users))
    baseline_scores = baseline.predict(val_repeat)
    mixed_scores = challenger.predict(val_mixed)
    baseline_p5 = ranking_metrics(val_repeat, val_sizes, baseline_scores, 5)["precision_at_5"]
    mixed_p5 = ranking_metrics(val_mixed, val_sizes, mixed_scores, 5)["precision_at_5"]
    difference = _hits(val_mixed, mixed_scores, validation_users) - _hits(
        val_repeat, baseline_scores, validation_users,
    )
    interval = _interval(difference)
    selected = ("mixed" if mixed_p5 - baseline_p5 >= .005 and interval[0] > 0
                else "repeat")
    print(f"Validation repeat={baseline_p5:.5f}, mixed={mixed_p5:.5f}; "
          f"selected={selected}", flush=True)
    novel = mixed.filter(pl.col("is_novel") == 1)
    val_novel = novel.filter(pl.col("user_id").is_in(validation_users))
    print("Fitting a separate new-product preview ranker...", flush=True)
    novel_challenger = fit(novel, train_users, MIXED_COLUMNS)
    val_fusion_scores = -val_novel["fusion_rank"].to_numpy().astype(np.float32)
    val_specialist_scores = novel_challenger.predict(val_novel)
    fusion_p5 = ranking_metrics(val_novel, val_sizes, val_fusion_scores, 5)["precision_at_5"]
    specialist_p5 = ranking_metrics(val_novel, val_sizes,
                                    val_specialist_scores, 5)["precision_at_5"]
    novel_interval = _interval(
        _hits(val_novel, val_specialist_scores, validation_users) -
        _hits(val_novel, val_fusion_scores, validation_users)
    )
    novel_selected = ("specialist" if specialist_p5 - fusion_p5 >= .005 and
                      novel_interval[0] > 0 else "fusion")
    print(f"New-only validation fusion={fusion_p5:.5f}, "
          f"specialist={specialist_p5:.5f}; selected={novel_selected}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if selected == "repeat":
        joblib.dump(challenger, output_dir / "mixed_preview_model.joblib")
    fit_users = np.concatenate((train_users, validation_users))
    final_frame = mixed if selected == "mixed" else repeat
    final_columns = MIXED_COLUMNS if selected == "mixed" else tuple(FEATURE_COLUMNS)
    final_model = fit(final_frame, fit_users, final_columns)
    joblib.dump(final_model, output_dir / "model.joblib")
    test_frame = final_frame.filter(pl.col("user_id").is_in(test_users))
    test_sizes = sizes.filter(pl.col("user_id").is_in(test_users))
    test_scores = final_model.predict(test_frame)
    test_metrics = ranking_metrics(test_frame, test_sizes, test_scores, 5)
    test_novel = novel.filter(pl.col("user_id").is_in(test_users))
    fusion_test_scores = -test_novel["fusion_rank"].to_numpy().astype(np.float32)
    if novel_selected == "specialist":
        final_novel = fit(novel, fit_users, MIXED_COLUMNS)
        joblib.dump(final_novel, output_dir / "novel_preview_model.joblib")
        novel_test_scores = final_novel.predict(test_novel)
    else:
        novel_test_scores = fusion_test_scores
    novel_test_metrics = ranking_metrics(test_novel, test_sizes, novel_test_scores, 5)
    rank_candidates(
        test_novel.select("user_id", "product_id", "product_name", "label"),
        novel_test_scores,
    ).filter(pl.col("rank") <= 5).write_csv(
        output_dir / "test_novel_preview.csv"
    )
    test_mixed = mixed.filter(pl.col("user_id").is_in(test_users))
    test_labels = test_mixed.filter(pl.col("is_novel") == 1)
    novel_targets = targets.filter(pl.col("user_id").is_in(test_users)).join(
        known, on=["user_id", "product_id"], how="anti",
    )
    selected_top = rank_candidates(
        test_frame.select("user_id", "product_id", "product_name", "label", "is_novel"),
        test_scores,
    ).filter(pl.col("rank") <= 5)
    selected_top.write_csv(output_dir / "test_recommendations.csv")
    report = {
        "decision_time": "before next order starts",
        "source_population": f"same {len(sampled)} original-training-user sample as retrieval",
        "train_users": len(train_users), "validation_users": len(validation_users),
        "test_users": len(test_users), "retrieval_variant": variant,
        "novel_budget": budget, "rounds": rounds,
        "candidate_rows": {"repeat": repeat.height, "mixed": mixed.height},
        "validation": {"repeat_p5": baseline_p5, "mixed_p5": mixed_p5,
                       "mixed_gain": mixed_p5 - baseline_p5,
                       "paired_95pct_interval": interval},
        "promotion_rule": "+0.005 absolute validation P@5 and positive paired interval",
        "selected": selected, "selected_test": test_metrics,
        "novel_preview_validation": {
            "fusion_p5": fusion_p5, "specialist_p5": specialist_p5,
            "specialist_gain": specialist_p5 - fusion_p5,
            "paired_95pct_interval": novel_interval,
        },
        "novel_preview_selected": novel_selected,
        "novel_preview_selected_test": novel_test_metrics,
        "novel_preview_fusion_test_p5": ranking_metrics(
            test_novel, test_sizes, fusion_test_scores, 5,
        )["precision_at_5"],
        "selected_test_novel_displayed": selected_top.filter(
            pl.col("is_novel") == 1).height,
        "selected_test_novel_hits": int(selected_top.filter(
            pl.col("is_novel") == 1)["label"].sum()),
        "test_novel_candidate_recall": float(test_labels["label"].sum() /
                                               novel_targets.height),
        "test_novel_target_items": novel_targets.height,
        "note": "The selected pilot model is not automatically promoted to the full-run CLI.",
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--retrieval-dir", type=Path,
                        default=Path("artifacts/discovery_10k_ablation"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/discovery_ranker_10k"))
    parser.add_argument("--max-users", type=int, default=10000)
    parser.add_argument("--rounds", type=int, default=350)
    args = parser.parse_args()
    run(args.data_dir, args.retrieval_dir, args.output_dir, args.max_users, args.rounds)
