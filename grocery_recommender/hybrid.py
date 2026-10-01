"""Repeat-first recommendations with a limited number of discovery items."""

import polars as pl


def combine_recommendations(repeat_ranked: pl.DataFrame, discovery_ranked: pl.DataFrame,
                            k: int = 5, min_probability: float = 0.5,
                            max_discovery: int = 2) -> pl.DataFrame:
    """Keep accepted repeats, then append unseen items in discovery rank order.

    repeat_ranked includes all previously purchased candidates, including those
    below the cutoff. Discovery scores never share the repeat probability field.
    """
    if k < 1:
        raise ValueError("k must be positive")
    if not 0 <= min_probability <= 1:
        raise ValueError("min_probability must be between zero and one")
    if max_discovery not in (0, 1, 2):
        raise ValueError("max_discovery must be zero, one, or two")

    repeats = (repeat_ranked.sort("user_id", "rank")
               .filter(pl.col("score") >= min_probability)
               .unique(subset=["user_id", "product_id"], maintain_order=True)
               .with_columns(pl.col("user_id").cum_count().over("user_id")
                             .alias("source_rank"))
               .filter(pl.col("source_rank") <= k))
    counts = repeats.group_by("user_id").agg(pl.len().alias("repeat_count"))
    unseen = (discovery_ranked.sort("user_id", "preview_rank")
              .join(repeat_ranked.select("user_id", "product_id").unique(),
                    on=["user_id", "product_id"], how="anti", maintain_order="left")
              .unique(subset=["user_id", "product_id"], maintain_order=True)
              .with_columns(pl.col("user_id").cum_count().over("user_id")
                            .alias("source_rank"))
              .join(counts, on="user_id", how="left", maintain_order="left")
              .with_columns(pl.col("repeat_count").fill_null(0))
              .filter(pl.col("source_rank") <= pl.min_horizontal(
                  pl.lit(max_discovery), k - pl.col("repeat_count"))))

    def display_columns(frame: pl.DataFrame, kind: str) -> pl.DataFrame:
        probability = (pl.col("score").cast(pl.Float64) if kind == "Repeat"
                       else pl.lit(None, dtype=pl.Float64))
        return frame.select(
            "user_id", "product_id", "product_name", "source_rank",
            pl.lit(kind).alias("recommendation_type"),
            probability.alias("estimated_reorder_probability"),
            pl.lit(0 if kind == "Repeat" else 1).alias("type_order"),
        )

    return (pl.concat([display_columns(repeats, "Repeat"),
                       display_columns(unseen, "Discovery")])
            .sort("user_id", "type_order", "source_rank")
            .with_columns(pl.col("user_id").cum_count().over("user_id").alias("rank"))
            .select("user_id", "rank", "recommendation_type", "product_id",
                    "product_name", "estimated_reorder_probability"))
