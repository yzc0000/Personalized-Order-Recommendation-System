"""Shared history-only features for pilot repeat and discovery ranking."""

import polars as pl

from .data import DataBundle
from .features import FEATURE_COLUMNS, FeatureSet
from .mixed import build_mixed_candidates


EXTRA_COLUMNS = ("neighbor_score", "basket_score", "popularity_score",
                 "content_score", "source_count", "fusion_rank")
MIXED_COLUMNS = tuple(FEATURE_COLUMNS) + (
    "is_novel", "anchor_aisle_count", "anchor_department_count",
) + EXTRA_COLUMNS


def build_discovery_candidates(
    bundle: DataBundle, base: FeatureSet, fused: pl.DataFrame,
    sources: dict[str, pl.DataFrame], neighbor_known: pl.DataFrame,
    anchors: pl.DataFrame, targets: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build the same candidate features for offline evaluation and serving."""
    known = base.candidates.select("user_id", "product_id")
    placeholder_known = known.with_columns(pl.lit(0.0).alias("basket_affinity"))
    placeholder_new = fused.select("user_id", "product_id").with_columns(
        pl.lit(0.0).alias("basket_affinity")
    )
    repeat, mixed = build_mixed_candidates(
        bundle, base, placeholder_new, placeholder_known, anchors, targets,
        exclude_anchors=False,
    )
    extra = (pl.concat([known, fused.select("user_id", "product_id")]).unique()
             .join(pl.concat([sources["neighbor"], neighbor_known]),
                   on=["user_id", "product_id"], how="left", maintain_order="left")
             .join(sources["basket"], on=["user_id", "product_id"], how="left",
                   maintain_order="left")
             .join(sources["popularity"], on=["user_id", "product_id"],
                   how="left", maintain_order="left"))
    if "content" in sources:
        extra = extra.join(sources["content"], on=["user_id", "product_id"],
                           how="left", maintain_order="left")
    else:
        extra = extra.with_columns(pl.lit(0.0).alias("content_score"))
    extra = extra.join(fused.select("user_id", "product_id",
                                    pl.col("rank").alias("fusion_rank")),
                       on=["user_id", "product_id"], how="left",
                       maintain_order="left")
    mixed = (mixed.join(extra, on=["user_id", "product_id"], how="left",
                        maintain_order="left")
             .with_columns(*[pl.col(c).fill_null(0) for c in
                             ("neighbor_score", "basket_score", "popularity_score",
                              "content_score", "fusion_rank")])
             .with_columns(
                 pl.when(pl.col("is_novel") == 1).then(
                     sum((pl.col(c) > 0).cast(pl.Int8) for c in
                         ("neighbor_score", "basket_score", "popularity_score",
                          "content_score"))
                 ).otherwise(0).alias("source_count")
             )
             .sort("user_id", "product_id"))
    repeat = repeat.sort("user_id", "product_id")
    if mixed.select(pl.struct("user_id", "product_id").n_unique()).item() != mixed.height:
        raise AssertionError("Duplicate candidate in fused discovery frame")
    return repeat, mixed
