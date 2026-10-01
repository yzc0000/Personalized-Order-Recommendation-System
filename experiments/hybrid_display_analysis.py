"""Evaluate repeat-first display rules using matched saved pilot predictions.

Repeat and novel scores are never compared with one another. Repeat items pass
their probability cutoff; unseen products fill the remaining display positions
in the specialist's saved rank order. No models are trained by this script.
"""

import argparse
import json
from pathlib import Path

import polars as pl

from grocery_recommender.hybrid import combine_recommendations


def run(ranker_dir: Path, retrieval_dir: Path, output_dir: Path,
        threshold: float = 0.5, k: int = 5) -> dict:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    if not 1 <= k <= 5:
        raise ValueError("saved prediction lists support k between one and five")
    model_report = json.loads((ranker_dir / "metrics.json").read_text())
    retrieval_report = json.loads((retrieval_dir / "metrics.json").read_text())
    if model_report["selected"] != "repeat":
        raise ValueError("This analysis requires saved repeat-only predictions")
    repeat = pl.read_csv(ranker_dir / "test_recommendations.csv")
    novel = pl.read_csv(ranker_dir / "test_novel_preview.csv")
    repeat = repeat.filter(pl.col("rank") <= k)
    novel = novel.filter(pl.col("rank") <= k)
    user_ids = repeat["user_id"].unique().sort()
    n_users = model_report["test_users"]
    if len(user_ids) != n_users or len(novel["user_id"].unique()) != n_users:
        raise ValueError("Saved lists do not cover the expected pilot customers")
    if set(user_ids.to_list()) != set(novel["user_id"].unique().to_list()):
        raise ValueError("Repeat and novel lists must cover the same customers")
    if repeat["is_novel"].sum() != 0:
        raise ValueError("Repeat list contains novel candidates")
    if repeat.join(novel, on=["user_id", "product_id"]).height:
        raise ValueError("Saved repeat and novel suggestions overlap")
    for frame in (repeat, novel):
        if frame.select("user_id", "product_id").unique().height != frame.height:
            raise ValueError("Saved predictions contain duplicate customer-product pairs")

    actual_total = retrieval_report["test"]["target_items"]
    actual_novel = retrieval_report["test"]["novel_target_items"]
    users = pl.DataFrame({"user_id": user_ids})
    retained = repeat.filter(pl.col("score") >= threshold)
    counts = users.join(retained.group_by("user_id").agg(
        pl.len().alias("repeat_count")), on="user_id", how="left"
    ).with_columns(pl.col("repeat_count").fill_null(0))

    def frame_with_origin(frame: pl.DataFrame, origin: str) -> pl.DataFrame:
        return frame.select("user_id", "product_id", "product_name", "label",
                            "score", "rank").with_columns(pl.lit(origin).alias("origin"))

    def metrics(frame: pl.DataFrame) -> dict:
        n_shown = frame.height
        hits = int(frame["label"].sum())
        r = frame.filter(pl.col("origin") == "repeat")
        n = frame.filter(pl.col("origin") == "novel")
        rh = int(r["label"].sum())
        nh = int(n["label"].sum())
        return {
            "suggestions_shown": n_shown,
            "matching_suggestions": hits,
            "precision_among_shown": hits / n_shown if n_shown else None,
            "matches_per_customer": hits / n_users,
            "shown_per_customer": n_shown / n_users,
            "precision_per_available_slot": hits / (k * n_users),
            "share_with_no_suggestions": 1 - frame["user_id"].n_unique() / n_users,
            "repeat_shown": r.height,
            "repeat_matches": rh,
            "repeat_precision_among_shown": rh / r.height if r.height else None,
            "novel_shown": n.height,
            "novel_matches": nh,
            "novel_precision_among_shown": nh / n.height if n.height else None,
            "novel_recall_micro": nh / actual_novel,
            "all_item_recall_micro": hits / actual_total,
        }

    unthresholded = frame_with_origin(repeat, "repeat")
    base = frame_with_origin(retained, "repeat")
    policies = {"repeat_top5_no_cutoff": metrics(unthresholded),
                "repeat_cutoff_only": metrics(base)}
    for name, max_new in (("fill_all_empty_slots", k), ("fill_up_to_two", 2)):
        selected_new = novel.join(counts, on="user_id").filter(
            pl.col("rank") <= pl.min_horizontal(k - pl.col("repeat_count"),
                                                pl.lit(max_new))
        )
        combined = pl.concat([base, frame_with_origin(selected_new, "novel")])
        policies[name] = metrics(combined)
        if name == "fill_all_empty_slots":
            fill_all_display = (combined.with_columns(
                (pl.col("origin") == "novel").cast(pl.Int8).alias("origin_order")
            ).sort("user_id", "origin_order", "rank")
                .with_columns(pl.col("user_id").cum_count().over("user_id")
                              .alias("display_rank"))
                .drop("origin_order"))

    # Use the runtime composition rule for the selected bounded policy too.
    display = combine_recommendations(
        repeat, novel.rename({"rank": "preview_rank"}), k, threshold, 2,
    ).join(pl.concat([repeat.select("user_id", "product_id", "label"),
                      novel.select("user_id", "product_id", "label")]),
           on=["user_id", "product_id"], how="left", maintain_order="left")
    policies["fill_up_to_two"] = metrics(display.with_columns(
        pl.when(pl.col("recommendation_type") == "Repeat")
        .then(pl.lit("repeat")).otherwise(pl.lit("novel")).alias("origin")))

    report = {
        "users": n_users, "model_rounds": model_report["rounds"],
        "repeat_probability_cutoff": threshold, "display_limit": k,
        "actual_next_order_items": actual_total,
        "actual_novel_items": actual_novel,
        "policies": policies,
        "selected_policy": "fill_up_to_two",
        "scope": "Same saved 1,500-customer discovery pilot test predictions. "
                 "This split has already been inspected during development.",
        "limitations": [
            "These results do not evaluate the full-data deployment reorder model.",
            "Saved top-five lists cannot evaluate more than five suggestions per source.",
            "Purchase matching does not measure whether recommendations cause purchases.",
            "The novel score has not been calibrated as a purchase probability.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n",
                                            encoding="utf-8")
    display.write_csv(output_dir / "test_hybrid_recommendations.csv")
    fill_all_display.write_csv(output_dir / "test_fill_all_recommendations.csv")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranker-dir", type=Path,
                        default=Path("artifacts/discovery_ranker_filled_10k"))
    parser.add_argument("--retrieval-dir", type=Path,
                        default=Path("artifacts/discovery_10k_ablation"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/hybrid_display_10k"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    run(args.ranker_dir, args.retrieval_dir, args.output_dir, args.threshold)


if __name__ == "__main__":
    main()
