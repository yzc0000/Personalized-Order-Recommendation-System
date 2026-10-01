"""Candidate retrieval for products a customer has not bought before."""

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix


def item_collaborative_candidates(
    known: pl.DataFrame, product_stats: pl.DataFrame, users: np.ndarray,
    top_products: int = 5000, per_user: int = 20,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Retrieve products owned by customers with overlapping product histories.

    Product similarity uses only prior purchases; target baskets are never read.
    The first result contains unseen products, the second scores known products.
    """
    if not 1 <= per_user < top_products:
        raise ValueError("Require 1 <= per_user < top_products")
    users = np.sort(np.unique(users))
    products = product_stats.sort(
        ["product_purchase_count", "product_id"], descending=[True, False]
    ).head(top_products)["product_id"].to_numpy()
    if len(products) <= per_user:
        raise ValueError("Not enough products for requested candidate count")
    user_index = {int(value): index for index, value in enumerate(users)}
    product_index = {int(value): index for index, value in enumerate(products)}
    observed = known.select("user_id", "product_id").unique().filter(
        pl.col("product_id").is_in(products)
    )
    rows = np.fromiter((user_index[int(value)] for value in observed["user_id"]),
                       dtype=np.int32, count=observed.height)
    columns = np.fromiter((product_index[int(value)] for value in observed["product_id"]),
                          dtype=np.int32, count=observed.height)
    incidence = csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(len(users), len(products)),
    )
    similarity = (incidence.T @ incidence).toarray().astype(np.float32)
    frequency = np.diag(similarity).copy()
    np.fill_diagonal(similarity, 0)
    similarity /= np.sqrt(np.maximum(frequency[:, None] * frequency[None, :], 1))
    scores = np.asarray(incidence @ similarity, dtype=np.float32)
    known_scores = pl.DataFrame({
        "user_id": users[rows],
        "product_id": products[columns],
        "collaborative_score": scores[rows, columns],
    })
    scores[rows, columns] = -np.inf
    top = np.argpartition(scores, -per_user, axis=1)[:, -per_user:]
    user_rows = np.repeat(np.arange(len(users)), per_user)
    product_columns = top.reshape(-1)
    candidate_scores = scores[user_rows, product_columns]
    keep = np.isfinite(candidate_scores) & (candidate_scores > 0)
    novel = pl.DataFrame({
        "user_id": users[user_rows[keep]],
        "product_id": products[product_columns[keep]],
        "collaborative_score": candidate_scores[keep],
    }).sort(["user_id", "collaborative_score", "product_id"],
            descending=[False, True, False])
    return novel, known_scores
