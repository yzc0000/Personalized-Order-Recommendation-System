"""Build one row per previously purchased user/product pair."""

from dataclasses import dataclass

import polars as pl

from .data import DataBundle


FEATURE_COLUMNS = [
    "user_order_count",
    "user_item_count",
    "user_unique_products",
    "user_average_basket_size",
    "user_mean_days_between_orders",
    "up_purchase_count",
    "up_reorder_count",
    "up_purchase_share",
    "up_order_gap",
    "up_days_since_last_purchase",
    "up_last_order_fraction",
    "up_mean_cart_position",
    "product_purchase_count",
    "product_reorder_rate",
    "user_department_share",
    "user_aisle_share",
    "up_recent3_count",
    "up_recent5_count",
    "up_order_rate_since_first",
    "up_mean_order_interval",
    "up_mean_days_interval",
    "up_next_order_gap_ratio",
]

# Pilot features are computed from completed orders only. Keep them separate
# until a user-held-out comparison justifies changing the deployed model.
SEQUENCE_PILOT_COLUMNS = [
    "up_last2_count",
    "up_recent10_count",
    "up_previous_order_gap",
    "up_order_interval_std",
    "up_last_cart_position",
    "up_recent5_trend",
]

REPLENISHMENT_PILOT_COLUMNS = [
    "up_trailing_streak", "up_recent3_order_rate",
    "up_aisle_purchase_fraction", "user_aisle_diversity",
]

CONTEXT_ONLY_COLUMNS = [
    "up_days_at_next_order",
    "up_next_days_interval_ratio",
    "next_days_since_prior_order",
    "next_order_dow",
    "next_order_hour_of_day",
    "up_last_dow_distance",
    "up_last_hour_distance",
]
CONTEXT_FEATURE_COLUMNS = FEATURE_COLUMNS + CONTEXT_ONLY_COLUMNS


@dataclass
class FeatureSet:
    candidates: pl.DataFrame
    target_sizes: pl.DataFrame
    target_item_count: int
    candidate_hits: int


def build_features(bundle: DataBundle, include_context: bool = False,
                   include_sequence_pilot: bool = False,
                   include_replenishment_pilot: bool = False) -> FeatureSet:
    """Build next-basket candidates; labels come only from target product rows.

    The default uses historical behavior available before the next order starts.
    include_context is for retrospective order-start experiments only.
    include_sequence_pilot adds optional history-only research features.
    """
    history_orders = (
        bundle.history_orders.sort("user_id", "order_number")
        .with_columns(
            pl.col("days_since_prior_order")
            .fill_null(0)
            .cum_sum()
            .over("user_id")
            .alias("elapsed_observed_days")
        )
    )
    user_orders = history_orders.group_by("user_id").agg(
        pl.col("order_number").max().alias("user_order_count"),
        pl.col("days_since_prior_order").mean().fill_null(0).alias("user_mean_days_between_orders"),
        pl.col("elapsed_observed_days").max().alias("user_elapsed_days"),
    )
    lines = (
        bundle.history_lines.join(
            history_orders.select(
                "order_id", "user_id", "order_number", "elapsed_observed_days",
                "order_dow", "order_hour_of_day",
            ),
            on="order_id", how="inner",
        )
        .join(user_orders.select("user_id", "user_order_count"), on="user_id", how="left")
        .join(bundle.products.select("product_id", "aisle_id", "department_id"),
              on="product_id", how="left")
    )
    user_items = lines.group_by("user_id").agg(
        pl.len().alias("user_item_count"),
        pl.col("product_id").n_unique().alias("user_unique_products"),
    )
    department = lines.group_by("user_id", "department_id").agg(
        pl.len().alias("user_department_items")
    )
    aisle_aggs = [pl.len().alias("user_aisle_items")]
    if include_replenishment_pilot:
        aisle_aggs.append(pl.col("product_id").n_unique().alias("user_aisle_diversity"))
    aisle = lines.group_by("user_id", "aisle_id").agg(aisle_aggs)
    user_product_aggs = [
        pl.len().alias("up_purchase_count"),
        pl.col("reordered").sum().alias("up_reorder_count"),
        pl.col("order_number").min().alias("up_first_order_number"),
        pl.col("order_number").max().alias("up_last_order_number"),
        pl.col("elapsed_observed_days").min().alias("up_first_elapsed_days"),
        pl.col("elapsed_observed_days").max().alias("up_last_elapsed_days"),
        pl.col("add_to_cart_order").mean().alias("up_mean_cart_position"),
        (pl.col("order_number") >= pl.col("user_order_count") - 2)
        .sum().alias("up_recent3_count"),
        (pl.col("order_number") >= pl.col("user_order_count") - 4)
        .sum().alias("up_recent5_count"),
        pl.col("order_dow").sort_by("order_number").last().alias("up_last_order_dow"),
        pl.col("order_hour_of_day").sort_by("order_number").last().alias("up_last_order_hour"),
    ]
    if include_sequence_pilot:
        user_product_aggs.extend([
            (pl.col("order_number") >= pl.col("user_order_count") - 1)
            .sum().alias("up_last2_count"),
            (pl.col("order_number") >= pl.col("user_order_count") - 9)
            .sum().alias("up_recent10_count"),
            pl.when(pl.len() > 1)
            .then(pl.col("order_number").sort().last() -
                  pl.col("order_number").sort().tail(2).first())
            .otherwise(0).alias("up_previous_order_gap"),
            pl.col("order_number").sort().diff().std().fill_null(0)
            .alias("up_order_interval_std"),
            pl.col("add_to_cart_order").sort_by("order_number").last()
            .alias("up_last_cart_position"),
        ])
    if include_replenishment_pilot:
        user_product_aggs.append(
            pl.col("order_number").sort(descending=True).diff().abs()
            .fill_null(1).eq(1).cum_prod().sum().alias("up_trailing_streak_raw")
        )
    user_product = lines.group_by("user_id", "product_id").agg(user_product_aggs)
    candidates = (
        user_product.join(user_orders, on="user_id", how="left")
        .join(user_items, on="user_id", how="left")
        .join(bundle.products.select("product_id", "product_name", "aisle_id", "department_id"),
              on="product_id", how="left")
        .join(bundle.product_stats, on="product_id", how="left")
        .join(department, on=["user_id", "department_id"], how="left")
        .join(aisle, on=["user_id", "aisle_id"], how="left")
        .with_columns(
            (pl.col("user_item_count") / pl.col("user_order_count")).alias("user_average_basket_size"),
            (pl.col("up_purchase_count") / pl.col("user_order_count")).alias("up_purchase_share"),
            (pl.col("user_order_count") - pl.col("up_last_order_number")).alias("up_order_gap"),
            (pl.col("user_elapsed_days") - pl.col("up_last_elapsed_days")).alias("up_days_since_last_purchase"),
            (pl.col("up_last_order_number") / pl.col("user_order_count")).alias("up_last_order_fraction"),
            (pl.col("user_department_items") / pl.col("user_item_count")).alias("user_department_share"),
            (pl.col("user_aisle_items") / pl.col("user_item_count")).alias("user_aisle_share"),
            (pl.col("up_purchase_count") /
             (pl.col("user_order_count") - pl.col("up_first_order_number") + 1))
            .alias("up_order_rate_since_first"),
            pl.when(pl.col("up_purchase_count") > 1)
            .then((pl.col("up_last_order_number") - pl.col("up_first_order_number")) /
                  (pl.col("up_purchase_count") - 1))
            .otherwise(0).alias("up_mean_order_interval"),
            pl.when(pl.col("up_purchase_count") > 1)
            .then((pl.col("up_last_elapsed_days") - pl.col("up_first_elapsed_days")) /
                  (pl.col("up_purchase_count") - 1))
            .otherwise(0).alias("up_mean_days_interval"),
        )
        .with_columns(
            pl.when(pl.col("up_mean_order_interval") > 0)
            .then((pl.col("up_order_gap") + 1) / pl.col("up_mean_order_interval"))
            .otherwise(0).alias("up_next_order_gap_ratio"),
        )
    )
    if include_sequence_pilot:
        candidates = candidates.with_columns(
            (pl.col("up_recent5_count") /
             pl.min_horizontal(pl.col("user_order_count"), pl.lit(5)) -
             pl.col("up_purchase_share")).alias("up_recent5_trend")
        )
    if include_replenishment_pilot:
        candidates = candidates.with_columns(
            pl.when(pl.col("up_order_gap") == 0)
            .then(pl.col("up_trailing_streak_raw"))
            .otherwise(0).alias("up_trailing_streak"),
            (pl.col("up_recent3_count") /
             pl.min_horizontal(pl.col("user_order_count"), pl.lit(3)))
            .alias("up_recent3_order_rate"),
            (pl.col("up_purchase_count") / pl.col("user_aisle_items"))
            .alias("up_aisle_purchase_fraction"),
        )
    if include_context:
        context_orders = (
            bundle.context_orders if bundle.context_orders is not None else bundle.target_orders
        )
        if context_orders.is_empty():
            raise ValueError("Next-order context is required for an order-start model")
        context = context_orders.select(
            "user_id",
            pl.col("order_dow").alias("next_order_dow"),
            pl.col("order_hour_of_day").alias("next_order_hour_of_day"),
            pl.col("days_since_prior_order").alias("next_days_since_prior_order"),
        )
        candidates = (
            candidates.join(context, on="user_id", how="left")
            .with_columns(
                (pl.col("user_elapsed_days") - pl.col("up_last_elapsed_days") +
                 pl.col("next_days_since_prior_order")).alias("up_days_at_next_order"),
                (pl.col("next_order_dow") - pl.col("up_last_order_dow")).abs()
                .alias("raw_dow_distance"),
                (pl.col("next_order_hour_of_day") - pl.col("up_last_order_hour")).abs()
                .alias("raw_hour_distance"),
            )
            .with_columns(
            pl.when(pl.col("up_mean_days_interval") > 0)
            .then(pl.col("up_days_at_next_order") / pl.col("up_mean_days_interval"))
            .otherwise(0).alias("up_next_days_interval_ratio"),
            pl.min_horizontal(pl.col("raw_dow_distance"), 7 - pl.col("raw_dow_distance"))
            .alias("up_last_dow_distance"),
            pl.min_horizontal(pl.col("raw_hour_distance"), 24 - pl.col("raw_hour_distance"))
            .alias("up_last_hour_distance"),
            )
        )
    if bundle.target_orders.is_empty():
        return FeatureSet(candidates, pl.DataFrame(schema={"user_id": pl.Int64, "target_size": pl.Int64}), 0, 0)

    targets = bundle.target_lines.join(
        bundle.target_orders.select("order_id", "user_id"), on="order_id", how="inner"
    ).select("user_id", "product_id", "reordered")
    target_sizes = targets.group_by("user_id").agg(pl.len().alias("target_size"))
    target_pairs = targets.select("user_id", "product_id").with_columns(pl.lit(1).alias("label"))
    candidates = candidates.join(target_pairs, on=["user_id", "product_id"], how="left").with_columns(
        pl.col("label").fill_null(0).cast(pl.Int8)
    )
    candidate_hits = int(candidates.select(pl.col("label").sum()).item())
    historical_target = targets.filter(pl.col("reordered") == 1).height
    if candidate_hits != historical_target:
        raise ValueError(
            f"Target reorder flag disagrees with candidate history: {candidate_hits} vs {historical_target}"
        )
    return FeatureSet(candidates, target_sizes, targets.height, candidate_hits)
