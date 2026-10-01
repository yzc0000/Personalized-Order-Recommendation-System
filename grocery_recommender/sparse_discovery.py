"""Catalog-wide sparse item relationships and simple content retrieval."""

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .data import DataBundle


def _positive_top(scores: np.ndarray, excluded: set[int],
                  product_ids: np.ndarray, count: int) -> list[tuple[int, float]]:
    scores = scores.copy()
    for pid in excluded:
        location = np.searchsorted(product_ids, pid)
        if location < len(product_ids) and product_ids[location] == pid:
            scores[location] = 0
    positive = np.flatnonzero(np.isfinite(scores) & (scores > 0))
    if len(positive) > count:
        positive = positive[np.argpartition(scores[positive], -count)[-count:]]
    return [(int(product_ids[j]), float(scores[j])) for j in
            sorted(positive, key=lambda j: (-scores[j], int(product_ids[j])))]


@dataclass
class SparseBasketIndex:
    product_ids: np.ndarray
    similarity: csr_matrix

    def score(self, anchors: list[int]) -> np.ndarray:
        locations = np.searchsorted(self.product_ids, anchors)
        locations = [int(j) for j, pid in zip(locations, anchors)
                     if j < len(self.product_ids) and self.product_ids[j] == pid]
        if not locations:
            return np.zeros(len(self.product_ids), dtype=np.float32)
        return np.asarray(self.similarity[locations].mean(axis=0),
                          dtype=np.float32).ravel()


def fit_sparse_basket_index(bundle: DataBundle, train_users: np.ndarray,
                            neighbors_per_product: int = 80,
                            min_pair_count: int = 3,
                            min_product_count: int = 5) -> SparseBasketIndex:
    """Learn sparse neighbors from completed training-user baskets only."""
    if min((neighbors_per_product, min_pair_count, min_product_count)) < 1:
        raise ValueError("All index settings must be positive")
    products = np.sort(bundle.products["product_id"].to_numpy().astype(np.int64))
    orders = bundle.history_orders.filter(pl.col("user_id").is_in(train_users))
    order_ids = np.sort(orders["order_id"].to_numpy().astype(np.int64))
    lines = bundle.history_lines.join(orders.select("order_id"), on="order_id")
    rows = np.searchsorted(order_ids, lines["order_id"].to_numpy())
    cols = np.searchsorted(products, lines["product_id"].to_numpy())
    matrix = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                        shape=(len(order_ids), len(products)))
    frequency = np.asarray(matrix.sum(axis=0)).ravel()
    eligible = np.flatnonzero(frequency >= min_product_count)
    small = matrix[:, eligible]
    co = (small.T @ small).tocsr()
    co.setdiag(0)
    co.eliminate_zeros()
    if co.nnz:
        keep = co.data >= min_pair_count
        co.data *= keep
        co.eliminate_zeros()
        small_frequency = frequency[eligible]
        # Normalized basket overlap, with shrinkage for rare products.
        for i in range(co.shape[0]):
            start, end = co.indptr[i:i + 2]
            if start == end:
                continue
            co.data[start:end] /= np.sqrt(
                (small_frequency[i] + 10) *
                (small_frequency[co.indices[start:end]] + 10)
            )
            if end - start > neighbors_per_product:
                values = co.data[start:end]
                discard = np.argpartition(values, -(neighbors_per_product))[
                    :len(values) - neighbors_per_product]
                values[discard] = 0
        co.eliminate_zeros()
    # Map the sparse trained-product graph back to the complete catalog.
    coordinate = co.tocoo()
    catalog_similarity = csr_matrix(
        (coordinate.data, (eligible[coordinate.row], eligible[coordinate.col])),
        shape=(len(products), len(products)), dtype=np.float32,
    )
    return SparseBasketIndex(products, catalog_similarity)


@dataclass
class ContentIndex:
    product_ids: np.ndarray
    vectors: csr_matrix
    vocabulary_size: int

    def score(self, anchors: list[int]) -> np.ndarray:
        positions = np.searchsorted(self.product_ids, anchors)
        positions = [int(j) for j, pid in zip(positions, anchors)
                     if j < len(self.product_ids) and self.product_ids[j] == pid]
        if not positions:
            return np.zeros(len(self.product_ids), dtype=np.float32)
        profile = normalize(np.asarray(self.vectors[positions].mean(axis=0),
                                       dtype=np.float32))
        return np.asarray(self.vectors @ profile.T, dtype=np.float32).ravel()


def fit_content_index(bundle: DataBundle) -> ContentIndex:
    """TF-IDF for names and supplied category names; no future behavior."""
    products = bundle.products.sort("product_id")
    if bundle.aisles is not None:
        products = products.join(bundle.aisles, on="aisle_id", how="left")
    if bundle.departments is not None:
        products = products.join(bundle.departments, on="department_id", how="left")
    columns = [c for c in ("product_name", "aisle", "department") if c in products.columns]
    texts = products.select(columns).fill_null("").to_dicts()
    corpus = [" ".join(str(row[c]) for c in columns) for row in texts]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                 max_features=150000, dtype=np.float32)
    vectors = vectorizer.fit_transform(corpus).tocsr()
    return ContentIndex(products["product_id"].to_numpy().astype(np.int64),
                        vectors, len(vectorizer.vocabulary_))


def retrieve_source(index: SparseBasketIndex | ContentIndex,
                    anchors: pl.DataFrame, known: pl.DataFrame,
                    per_user: int = 40, source: str = "basket"
                    ) -> pl.DataFrame:
    """Recommend unseen products from one source with deterministic ties."""
    anchor_map = dict(anchors.group_by("user_id").agg("product_id").iter_rows())
    known_map = dict(known.group_by("user_id").agg("product_id").iter_rows())
    rows = []
    for uid, items in anchor_map.items():
        scores = index.score(items)
        for pid, score in _positive_top(scores, set(known_map.get(uid, [])),
                                        index.product_ids, per_user):
            rows.append((int(uid), pid, score))
    return pl.DataFrame(rows, orient="row", schema={
        "user_id": pl.Int64, "product_id": pl.Int64,
        f"{source}_score": pl.Float32,
    })


def retrieve_popular(bundle: DataBundle, fit_users: np.ndarray,
                     known: pl.DataFrame, per_user: int = 40) -> pl.DataFrame:
    """Popularity from exactly the fitted customers' completed histories."""
    return retrieve_popular_index(fit_popularity_index(bundle, fit_users),
                                  known, per_user)


@dataclass
class PopularityIndex:
    product_ids: np.ndarray
    frequency: np.ndarray


def fit_popularity_index(bundle: DataBundle, fit_users: np.ndarray) -> PopularityIndex:
    """Saveable popularity list using only fitted users' completed orders."""
    orders = bundle.history_orders.filter(pl.col("user_id").is_in(fit_users))
    popularity = (bundle.history_lines.join(orders.select("order_id"), on="order_id")
                  .group_by("product_id").agg(pl.len().alias("frequency"))
                  .sort(["frequency", "product_id"], descending=[True, False]))
    return PopularityIndex(popularity["product_id"].to_numpy(),
                           popularity["frequency"].to_numpy())


def retrieve_popular_index(index: PopularityIndex, known: pl.DataFrame,
                           per_user: int = 40) -> pl.DataFrame:
    if per_user < 1:
        raise ValueError("per_user must be positive")
    known_map = dict(known.group_by("user_id").agg("product_id").iter_rows())
    rows = []
    for uid, owned_items in known_map.items():
        owned = set(owned_items)
        count = 0
        for pid, frequency in zip(index.product_ids, index.frequency):
            if pid not in owned:
                rows.append((int(uid), int(pid), float(frequency)))
                count += 1
                if count == per_user:
                    break
    return pl.DataFrame(rows, orient="row", schema={
        "user_id": pl.Int64, "product_id": pl.Int64,
        "popularity_score": pl.Float32,
    })
