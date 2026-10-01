"""Product replenishment priors estimated from training customers' prior orders."""

import numpy as np
import polars as pl

from .data import DataBundle


PRODUCT_PRIOR_COLUMNS = (
    "product_next_order_rate", "product_one_shot_fraction",
    "product_observed_transitions", "product_eligible_buyers",
)


def fit_product_priors(bundle: DataBundle, fit_users: np.ndarray) -> pl.DataFrame:
    """Estimate follow-up behavior without reading anyone's target basket.

    Last completed orders have no observed subsequent *prior* basket, so they
    are excluded from the next-order rate denominator. The one-shot statistic
    requires at least two following observed orders after first purchase.
    """
    orders = bundle.history_orders.filter(pl.col("user_id").is_in(fit_users))
    latest = orders.group_by("user_id").agg(
        pl.col("order_number").max().alias("last_order_number")
    )
    lines = bundle.history_lines.join(
        orders.select("order_id", "user_id", "order_number"), on="order_id"
    ).join(latest, on="user_id")
    next_pairs = lines.select(
        "user_id", "product_id",
        (pl.col("order_number") - 1).alias("order_number"),
    ).with_columns(pl.lit(1).alias("bought_next"))
    transitions = (
        lines.filter(pl.col("order_number") < pl.col("last_order_number"))
        .join(next_pairs, on=["user_id", "product_id", "order_number"], how="left")
        .group_by("product_id").agg(
            pl.len().alias("product_observed_transitions"),
            pl.col("bought_next").fill_null(0).sum().alias("next_order_repeats"),
        )
        .with_columns(
            ((pl.col("next_order_repeats") + 1) /
             (pl.col("product_observed_transitions") + 10))
            .alias("product_next_order_rate")
        )
    )
    buyer_events = (
        lines.group_by("user_id", "product_id").agg(
            pl.col("order_number").min().alias("first_order_number"),
            pl.len().alias("purchase_count"),
            pl.col("last_order_number").first(),
        )
        .filter(pl.col("first_order_number") <= pl.col("last_order_number") - 2)
        .group_by("product_id").agg(
            pl.len().alias("product_eligible_buyers"),
            (pl.col("purchase_count") == 1).sum().alias("one_shot_buyers"),
        )
        .with_columns(
            ((pl.col("one_shot_buyers") + 1) /
             (pl.col("product_eligible_buyers") + 2))
            .alias("product_one_shot_fraction")
        )
    )
    return (
        bundle.products.select("product_id")
        .join(transitions.select("product_id", "product_observed_transitions",
                                 "product_next_order_rate"),
              on="product_id", how="left")
        .join(buyer_events.select("product_id", "product_eligible_buyers",
                                  "product_one_shot_fraction"),
              on="product_id", how="left")
        .with_columns(
            pl.col("product_observed_transitions").fill_null(0),
            pl.col("product_next_order_rate").fill_null(0.1),
            pl.col("product_eligible_buyers").fill_null(0),
            pl.col("product_one_shot_fraction").fill_null(0.5),
        )
    )
