"""Load the Instacart source files and check their modeling grain."""

from dataclasses import dataclass
from pathlib import Path

import polars as pl


REQUIRED_FILES = (
    "orders.csv",
    "order_products__prior.csv",
    "order_products__train.csv",
    "products.csv",
    "aisles.csv",
    "departments.csv",
)


@dataclass
class DataBundle:
    history_orders: pl.DataFrame
    history_lines: pl.DataFrame
    target_orders: pl.DataFrame
    target_lines: pl.DataFrame
    products: pl.DataFrame
    product_stats: pl.DataFrame
    aisles: pl.DataFrame | None = None
    departments: pl.DataFrame | None = None
    context_orders: pl.DataFrame | None = None


@dataclass
class PreparedData:
    orders: pl.DataFrame
    products: pl.DataFrame
    aisles: pl.DataFrame
    departments: pl.DataFrame
    product_stats: pl.DataFrame


def prepare_data(data_dir: Path) -> PreparedData:
    """Load small shared tables and global prior statistics once for batches."""
    require_files(data_dir)
    product_stats = (
        pl.scan_csv(data_dir / "order_products__prior.csv")
        .group_by("product_id")
        .agg(
            pl.len().alias("product_purchase_count"),
            pl.col("reordered").mean().alias("product_reorder_rate"),
        )
        .collect(engine="streaming")
    )
    return PreparedData(
        pl.read_csv(data_dir / "orders.csv"),
        pl.read_csv(data_dir / "products.csv"),
        pl.read_csv(data_dir / "aisles.csv"),
        pl.read_csv(data_dir / "departments.csv"),
        product_stats,
    )


def require_files(data_dir: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source files in {data_dir}: {', '.join(missing)}")


def source_profile(data_dir: Path) -> dict:
    """Count the complete supplied order splits, beyond any sampled model users."""
    require_files(data_dir)
    splits = (
        pl.scan_csv(data_dir / "orders.csv")
        .group_by("eval_set")
        .agg(pl.len().alias("orders"), pl.col("user_id").n_unique().alias("users"))
        .collect(engine="streaming")
    )
    return {
        "order_splits": {
            row["eval_set"]: {"orders": row["orders"], "users": row["users"]}
            for row in splits.to_dicts()
        },
        "prior_order_items": pl.scan_csv(data_dir / "order_products__prior.csv")
        .select(pl.len()).collect(engine="streaming").item(),
        "train_order_items": pl.scan_csv(data_dir / "order_products__train.csv")
        .select(pl.len()).collect(engine="streaming").item(),
    }


def _lines_for_orders(path: Path, order_ids: pl.DataFrame) -> pl.DataFrame:
    return (
        pl.scan_csv(path)
        .join(order_ids.lazy(), on="order_id", how="semi")
        .collect(engine="streaming")
    )


def load_bundle(
    data_dir: Path, max_users: int | None = None, user_id: int | None = None,
    user_ids: list[int] | None = None, prepared: PreparedData | None = None,
) -> DataBundle:
    """Read selected user histories; product statistics always use all prior rows.

    max_users selects a repeatable random sample of labeled users.
    user_id loads one user's history for prediction without requiring a labeled order.
    """
    require_files(data_dir)
    if max_users is not None and max_users < 1:
        raise ValueError("max_users must be positive")
    if sum(x is not None for x in (max_users, user_id, user_ids)) > 1:
        raise ValueError("Choose one of max_users, user_id, or user_ids")

    prepared = prepared or prepare_data(data_dir)
    orders = prepared.orders
    if user_id is not None:
        selected = orders.filter(pl.col("user_id") == user_id)
        if selected.is_empty():
            raise ValueError(f"Unknown user_id: {user_id}")
    elif user_ids is not None:
        selected = orders.filter(pl.col("user_id").is_in(user_ids))
        if selected.select(pl.col("user_id").n_unique()).item() != len(set(user_ids)):
            raise ValueError("Some requested users do not exist")
    else:
        eligible = orders.filter(pl.col("eval_set") == "train").select("user_id")
        if max_users is not None:
            eligible = eligible.sample(n=min(max_users, eligible.height), seed=42)
        selected = orders.join(eligible, on="user_id", how="semi")

    history_orders = selected.filter(pl.col("eval_set") == "prior")
    target_orders = selected.filter(pl.col("eval_set") == "train")
    context_orders = selected.filter(pl.col("eval_set").is_in(["train", "test"]))
    history_lines = _lines_for_orders(
        data_dir / "order_products__prior.csv", history_orders.select("order_id")
    )
    target_lines = (
        _lines_for_orders(
            data_dir / "order_products__train.csv", target_orders.select("order_id")
        )
        if not target_orders.is_empty()
        else pl.DataFrame(schema={"order_id": pl.Int64, "product_id": pl.Int64,
                                  "add_to_cart_order": pl.Int64, "reordered": pl.Int64})
    )
    products = prepared.products
    aisles = prepared.aisles
    departments = prepared.departments
    product_stats = prepared.product_stats
    return DataBundle(
        history_orders, history_lines, target_orders, target_lines,
        products, product_stats, aisles, departments, context_orders,
    )


def validate_bundle(bundle: DataBundle) -> dict[str, int | float | None]:
    """Fail on broken keys or history/target joins; summarize selected population."""
    h, t, hp, tp, products = (
        bundle.history_orders, bundle.target_orders, bundle.history_lines,
        bundle.target_lines, bundle.products,
    )
    if h.is_empty() or hp.is_empty():
        raise ValueError("Selected users have no historical orders or products")
    if h.select(pl.col("order_id").n_unique()).item() != h.height:
        raise ValueError("Duplicate historical order_id")
    if pl.concat([h.select("order_id"), t.select("order_id")]).select(
        pl.col("order_id").n_unique()
    ).item() != h.height + t.height:
        raise ValueError("Duplicate order_id across history and target")
    if h.select(pl.struct("user_id", "order_number").n_unique()).item() != h.height:
        raise ValueError("Duplicate user/order_number in history")
    if hp.select(pl.struct("order_id", "product_id").n_unique()).item() != hp.height:
        raise ValueError("Duplicate order/product in history")
    if tp.height and tp.select(pl.struct("order_id", "product_id").n_unique()).item() != tp.height:
        raise ValueError("Duplicate order/product in target")
    if hp.join(h.select("order_id"), on="order_id", how="anti").height:
        raise ValueError("Historical product rows without an order")
    if tp.join(t.select("order_id"), on="order_id", how="anti").height:
        raise ValueError("Target product rows without an order")
    if hp.join(products.select("product_id"), on="product_id", how="anti").height:
        raise ValueError("Historical product rows without product metadata")
    if tp.join(products.select("product_id"), on="product_id", how="anti").height:
        raise ValueError("Target product rows without product metadata")
    if t.height and t.group_by("user_id").len().select(pl.col("len").max()).item() != 1:
        raise ValueError("More than one labeled next order per user")
    sequence = h.group_by("user_id").agg(
        pl.col("order_number").min().alias("first_order"),
        pl.col("order_number").max().alias("last_order"),
        pl.len().alias("order_count"),
    )
    if sequence.filter((pl.col("first_order") != 1) |
                       (pl.col("last_order") != pl.col("order_count"))).height:
        raise ValueError("Historical order numbers must start at 1 and be consecutive")
    if t.join(sequence.select("user_id"), on="user_id", how="anti").height:
        raise ValueError("Target users without historical orders")
    if t.height and t.join(sequence.select("user_id", "last_order"), on="user_id").filter(
        pl.col("order_number") != pl.col("last_order") + 1
    ).height:
        raise ValueError("Target order must follow the historical orders")
    if bundle.context_orders is not None:
        context = bundle.context_orders
        if context.group_by("user_id").len().select(pl.col("len").max()).item() != 1:
            raise ValueError("Expected one next-order context per selected user")
        if context.join(sequence.select("user_id", "last_order"), on="user_id").filter(
            pl.col("order_number") != pl.col("last_order") + 1
        ).height:
            raise ValueError("Next-order context must follow the historical orders")
    if h.filter(
        (pl.col("order_number") == 1) != pl.col("days_since_prior_order").is_null()
    ).height:
        raise ValueError("Only first orders should have missing days_since_prior_order")
    if h.filter(
        pl.col("days_since_prior_order").is_not_null() &
        ~pl.col("days_since_prior_order").is_between(0, 30)
    ).height:
        raise ValueError("days_since_prior_order outside expected 0–30 range")
    if hp.filter(~pl.col("reordered").is_in([0, 1])).height or tp.filter(~pl.col("reordered").is_in([0, 1])).height:
        raise ValueError("reordered must be binary")
    if products.select(pl.col("product_id").n_unique()).item() != products.height:
        raise ValueError("Duplicate product metadata key")
    if bundle.product_stats.join(products.select("product_id"), on="product_id", how="anti").height:
        raise ValueError("Global prior products without metadata")
    if bundle.aisles is not None:
        if bundle.aisles.select(pl.col("aisle_id").n_unique()).item() != bundle.aisles.height:
            raise ValueError("Duplicate aisle key")
        if products.join(bundle.aisles.select("aisle_id"), on="aisle_id", how="anti").height:
            raise ValueError("Products without aisle metadata")
    if bundle.departments is not None:
        if bundle.departments.select(pl.col("department_id").n_unique()).item() != bundle.departments.height:
            raise ValueError("Duplicate department key")
        if products.join(bundle.departments.select("department_id"), on="department_id", how="anti").height:
            raise ValueError("Products without department metadata")
    return {
        "users": h.select("user_id").n_unique(),
        "historical_orders": h.height,
        "historical_items": hp.height,
        "target_orders": t.height,
        "target_items": tp.height,
        "products": products.height,
        "aisles": bundle.aisles.height if bundle.aisles is not None else None,
        "departments": bundle.departments.height if bundle.departments is not None else None,
        "historical_missing_days_rate": h.filter(pl.col("days_since_prior_order").is_null()).height / h.height,
        "historical_average_basket_size": hp.height / h.height,
        "target_average_basket_size": tp.height / t.height if t.height else None,
    }
