"""Create leakage-controlled training examples from historical order prefixes."""

import numpy as np
import polars as pl

from .data import DataBundle
from .features import build_features


def historical_training_examples(
    bundle: DataBundle, user_ids: np.ndarray, snapshots: int
) -> pl.DataFrame:
    """Use the last `snapshots` prior orders as pseudo-targets for given users.

    Each pseudo-target sees only earlier orders from the same user. Callers must
    exclude global product statistics from the model feature list because those
    statistics include transactions after a historical pseudo-target.
    """
    if snapshots < 0:
        raise ValueError("snapshots must be nonnegative")
    if snapshots == 0:
        raise ValueError("Request at least one snapshot")
    orders = bundle.history_orders.filter(pl.col("user_id").is_in(user_ids))
    last = orders.group_by("user_id").agg(
        pl.col("order_number").max().alias("last_prior_order")
    )
    frames = []
    for lag in range(snapshots):
        target_orders = (
            orders.join(last, on="user_id")
            .filter((pl.col("order_number") == pl.col("last_prior_order") - lag) &
                    (pl.col("order_number") >= 3))
            .drop("last_prior_order")
        )
        if target_orders.is_empty():
            continue
        history_orders = (
            orders.join(
                target_orders.select("user_id", pl.col("order_number").alias("cutoff")),
                on="user_id",
            )
            .filter(pl.col("order_number") < pl.col("cutoff"))
            .drop("cutoff")
        )
        history_lines = bundle.history_lines.join(
            history_orders.select("order_id"), on="order_id", how="semi"
        )
        target_lines = bundle.history_lines.join(
            target_orders.select("order_id"), on="order_id", how="semi"
        )
        snapshot_bundle = DataBundle(
            history_orders=history_orders,
            history_lines=history_lines,
            target_orders=target_orders,
            target_lines=target_lines,
            products=bundle.products,
            product_stats=bundle.product_stats,
            aisles=bundle.aisles,
            departments=bundle.departments,
            context_orders=target_orders,
        )
        frames.append(
            build_features(snapshot_bundle).candidates.join(
                target_orders.select("user_id",
                                     pl.col("order_number").alias("snapshot_id")),
                on="user_id", how="left", maintain_order="left",
            )
        )
    if not frames:
        raise ValueError("No users had enough prior orders for historical snapshots")
    return pl.concat(frames, how="vertical").sort("user_id", "snapshot_id",
                                                     "product_id")
