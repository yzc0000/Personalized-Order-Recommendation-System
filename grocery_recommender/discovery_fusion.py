"""Deterministic, deduplicated fusion of unseen-product retrieval sources."""

import polars as pl


def rank_source(frame: pl.DataFrame, name: str, priority: int) -> pl.DataFrame:
    score_name = next(c for c in frame.columns if c.endswith("_score"))
    return (frame.sort(["user_id", score_name, "product_id"],
                       descending=[False, True, False])
            .with_columns(pl.col("user_id").cum_count().over("user_id").alias("source_rank"),
                          pl.lit(name).alias("source"),
                          pl.lit(priority).alias("source_priority"))
            .select("user_id", "product_id", "source_rank", "source",
                    "source_priority", pl.col(score_name).alias("source_score")))


def fuse_sources(sources: list[pl.DataFrame], budget: int) -> pl.DataFrame:
    """Round-robin source ranks, then remove duplicate products per user."""
    if budget < 1 or not sources:
        raise ValueError("Use at least one source and a positive budget")
    all_rows = pl.concat(sources, how="vertical")
    return (all_rows.sort(["user_id", "source_rank", "source_priority", "product_id"])
            .unique(subset=["user_id", "product_id"], keep="first",
                    maintain_order=True)
            .with_columns(pl.col("user_id").cum_count().over("user_id").alias("rank"))
            .filter(pl.col("rank") <= budget))
