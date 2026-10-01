"""Order-level product relationships learned from completed prior baskets."""

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from .data import DataBundle


@dataclass
class CoBasketIndex:
    product_ids: np.ndarray
    similarity: np.ndarray
    positions: dict[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.positions = {int(pid): position for position, pid in enumerate(self.product_ids)}

    def scores(self, anchors: list[int]) -> np.ndarray:
        rows = [self.positions[pid] for pid in set(anchors) if pid in self.positions]
        if not rows:
            return np.zeros(len(self.product_ids), dtype=np.float32)
        return self.similarity[rows].mean(axis=0).astype(np.float32)


def fit_co_basket_index(bundle: DataBundle, train_users: np.ndarray,
                        top_products: int = 3000) -> CoBasketIndex:
    """Fit only on prior orders of training users, never target baskets."""
    if top_products < 2:
        raise ValueError("top_products must be at least 2")
    orders = (
        bundle.history_orders.filter(pl.col("user_id").is_in(train_users))
        .select("order_id").sort("order_id")
        .with_row_index("order_position")
    )
    lines = bundle.history_lines.join(orders, on="order_id", how="inner")
    products = (
        lines.group_by("product_id").len()
        .sort(["len", "product_id"], descending=[True, False])
        .head(top_products).select("product_id")
        .with_row_index("product_position")
    )
    observed = lines.join(products, on="product_id", how="inner")
    matrix = csr_matrix(
        (np.ones(observed.height, dtype=np.float32),
         (observed["order_position"].to_numpy(),
          observed["product_position"].to_numpy())),
        shape=(orders.height, products.height),
    )
    cooccurrence = (matrix.T @ matrix).toarray().astype(np.float32)
    frequency = np.diag(cooccurrence).copy()
    np.fill_diagonal(cooccurrence, 0)
    # Shrink rare products to avoid chance one-basket matches dominating.
    similarity = cooccurrence / np.sqrt(
        (frequency[:, None] + 10) * (frequency[None, :] + 10)
    )
    return CoBasketIndex(products["product_id"].to_numpy(), similarity)


def retrieve_co_basket_candidates(
    index: CoBasketIndex, anchors: pl.DataFrame, known: pl.DataFrame,
    per_user: int = 40,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Score known products and retrieve unseen ones from available anchors."""
    if per_user < 1 or per_user >= len(index.product_ids):
        raise ValueError("per_user must be between 1 and the index size")
    anchor_map = defaultdict(list)
    for user_id, product_id in anchors.select("user_id", "product_id").iter_rows():
        anchor_map[int(user_id)].append(int(product_id))
    known_map = defaultdict(list)
    for user_id, product_id in known.select("user_id", "product_id").iter_rows():
        known_map[int(user_id)].append(int(product_id))
    novel_rows, known_rows = [], []
    for user_id, products in anchor_map.items():
        scores = index.scores(products)
        owned = known_map[user_id]
        for product_id in owned:
            position = index.positions.get(product_id)
            if position is not None:
                known_rows.append((user_id, product_id, float(scores[position])))
                scores[position] = -np.inf
        for product_id in products:
            position = index.positions.get(product_id)
            if position is not None:
                scores[position] = -np.inf
        if not np.isfinite(scores).any():
            continue
        top = np.argpartition(scores, -per_user)[-per_user:]
        top = sorted((int(position) for position in top if scores[position] > 0),
                     key=lambda position: (-scores[position], int(index.product_ids[position])))
        novel_rows.extend(
            (user_id, int(index.product_ids[position]), float(scores[position]))
            for position in top
        )
    schema = {"user_id": pl.Int64, "product_id": pl.Int64,
              "basket_affinity": pl.Float32}
    novel = pl.DataFrame(novel_rows, schema=schema, orient="row")
    known_scores = pl.DataFrame(known_rows, schema=schema, orient="row")
    return novel, known_scores
