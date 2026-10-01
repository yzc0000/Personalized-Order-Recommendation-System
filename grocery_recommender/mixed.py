"""Comparable repeat and new-product features for basket-aware ranking."""

import polars as pl

from .data import DataBundle
from .features import FEATURE_COLUMNS, FeatureSet


BASKET_FEATURE_COLUMNS = tuple(FEATURE_COLUMNS) + (
    "basket_affinity", "is_novel", "anchor_aisle_count", "anchor_department_count",
)


def build_mixed_candidates(
    bundle: DataBundle, base: FeatureSet, novel: pl.DataFrame,
    known_scores: pl.DataFrame, anchors: pl.DataFrame,
    remaining_targets: pl.DataFrame | None = None,
    exclude_anchors: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Construct labels only from unobserved target items.

    `anchors` are visible cart products in a session or the last completed
    basket before a future order. They cannot contain hidden target items.
    """
    labels = (
        remaining_targets.select("user_id", "product_id")
        if remaining_targets is not None else
        pl.DataFrame(schema={"user_id": pl.Int64, "product_id": pl.Int64})
    )
    labels = labels.unique().with_columns(pl.lit(1).cast(pl.Int8).alias("label"))
    known = (
        base.candidates.drop("label", strict=False)
        .join(known_scores, on=["user_id", "product_id"], how="left")
        .join(labels, on=["user_id", "product_id"], how="left")
        .with_columns(
            pl.col("basket_affinity").fill_null(0),
            pl.col("label").fill_null(0).cast(pl.Int8),
            pl.lit(0).cast(pl.Int8).alias("is_novel"),
        )
    )
    user_columns = [
        "user_order_count", "user_item_count", "user_unique_products",
        "user_average_basket_size", "user_mean_days_between_orders",
    ]
    user_info = known.group_by("user_id").agg(
        *[pl.col(name).first().alias(name) for name in user_columns]
    )
    history = (
        bundle.history_lines.join(bundle.history_orders.select("order_id", "user_id"),
                                  on="order_id")
        .join(bundle.products.select("product_id", "department_id", "aisle_id"),
              on="product_id")
    )
    department = history.group_by("user_id", "department_id").agg(
        pl.len().alias("department_items")
    )
    aisle = history.group_by("user_id", "aisle_id").agg(
        pl.len().alias("aisle_items")
    )
    new = (
        novel.join(bundle.products.select("product_id", "product_name", "department_id",
                                          "aisle_id"), on="product_id")
        .join(bundle.product_stats, on="product_id", how="left")
        .join(user_info, on="user_id")
        .join(department, on=["user_id", "department_id"], how="left")
        .join(aisle, on=["user_id", "aisle_id"], how="left")
        .join(labels, on=["user_id", "product_id"], how="left")
        .with_columns(
            (pl.col("department_items") / pl.col("user_item_count"))
            .fill_null(0).alias("user_department_share"),
            (pl.col("aisle_items") / pl.col("user_item_count"))
            .fill_null(0).alias("user_aisle_share"),
            pl.col("label").fill_null(0).cast(pl.Int8),
            pl.lit(1).cast(pl.Int8).alias("is_novel"),
        )
    )
    for name in FEATURE_COLUMNS:
        if name not in new.columns:
            new = new.with_columns(pl.lit(None).cast(pl.Float64).alias(name))

    anchor_categories = anchors.join(
        bundle.products.select("product_id", "aisle_id", "department_id"),
        on="product_id", how="left",
    )
    anchor_aisle = anchor_categories.group_by("user_id", "aisle_id").agg(
        pl.len().alias("anchor_aisle_count")
    )
    anchor_department = anchor_categories.group_by("user_id", "department_id").agg(
        pl.len().alias("anchor_department_count")
    )
    columns = ["user_id", "product_id", "product_name", "label", *BASKET_FEATURE_COLUMNS]

    def finish(frame: pl.DataFrame) -> pl.DataFrame:
        return (
            frame.join(anchor_aisle, on=["user_id", "aisle_id"], how="left")
            .join(anchor_department, on=["user_id", "department_id"], how="left")
            .with_columns(
                pl.col("anchor_aisle_count").fill_null(0),
                pl.col("anchor_department_count").fill_null(0),
            )
            .select(columns)
        )

    repeat_frame = finish(known)
    new_frame = finish(new)
    if exclude_anchors:
        observed = anchors.select("user_id", "product_id").unique()
        repeat_frame = repeat_frame.join(observed, on=["user_id", "product_id"],
                                         how="anti")
        new_frame = new_frame.join(observed, on=["user_id", "product_id"],
                                   how="anti")
    mixed = pl.concat([repeat_frame, new_frame], how="vertical_relaxed")
    return (repeat_frame.sort("user_id", "product_id"),
            mixed.sort("user_id", "product_id"))
