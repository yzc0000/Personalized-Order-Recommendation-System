"""Sparse, time-weighted customer neighbors for next-basket candidates.

This is a TIFU-KNN-inspired implementation. It groups historical baskets,
discounts older baskets and older groups, and mixes personal frequencies with
similar-customer frequencies. It is not a claim of reproducing the paper's
published preprocessing or hyperparameters exactly.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from .data import DataBundle


def _profiles(bundle: DataBundle, users: np.ndarray, products: np.ndarray,
              group_size: int, within_decay: float, group_decay: float) -> csr_matrix:
    users = np.asarray(users, dtype=np.int64)
    products = np.asarray(products, dtype=np.int64)
    if not np.all(users[:-1] < users[1:]) or not np.all(products[:-1] < products[1:]):
        raise ValueError("User and product IDs must be strictly increasing")
    history = bundle.history_orders.filter(pl.col("user_id").is_in(users))
    latest = history.group_by("user_id").agg(
        pl.col("order_number").max().alias("user_order_count")
    )
    lines = (
        bundle.history_lines.join(history.select("order_id", "user_id", "order_number"),
                                  on="order_id")
        .join(latest, on="user_id")
        .select("user_id", "product_id", "order_number", "user_order_count")
    )
    uid = lines["user_id"].to_numpy().astype(np.int64)
    pid = lines["product_id"].to_numpy().astype(np.int64)
    order = lines["order_number"].to_numpy().astype(np.int32)
    last = lines["user_order_count"].to_numpy().astype(np.int32)
    row = np.searchsorted(users, uid)
    col = np.searchsorted(products, pid)
    if np.any(row >= len(users)) or np.any(users[row] != uid):
        raise ValueError("Unexpected user ID in history")
    if np.any(col >= len(products)) or np.any(products[col] != pid):
        raise ValueError("Unexpected product ID in history")
    group = (order - 1) // group_size
    last_group = (last - 1) // group_size
    group_end = np.minimum((group + 1) * group_size, last)
    weights = (np.power(within_decay, group_end - order) *
               np.power(group_decay, last_group - group)).astype(np.float32)
    matrix = csr_matrix((weights, (row, col)),
                        shape=(len(users), len(products)), dtype=np.float32)
    # Divide by the total basket weight, leaving a weighted purchase rate.
    basket_weights = np.zeros(len(users), dtype=np.float32)
    for user_id, order_number, count in history.join(latest, on="user_id").select(
            "user_id", "order_number", "user_order_count").iter_rows():
        idx = np.searchsorted(users, user_id)
        period = (order_number - 1) // group_size
        end = min((period + 1) * group_size, count)
        basket_weights[idx] += (within_decay ** (end - order_number) *
                                group_decay ** (((count - 1) // group_size) - period))
    basket_weights = np.maximum(basket_weights, 1e-6)
    return matrix.multiply((1 / basket_weights)[:, None]).tocsr()


@dataclass
class TemporalNeighborIndex:
    product_ids: np.ndarray
    train_users: np.ndarray
    train_profiles: csr_matrix
    group_size: int = 7
    within_decay: float = 0.9
    group_decay: float = 0.7
    neighbors: int = 50
    personal_weight: float = 0.7

    def score_users(self, bundle: DataBundle, users: np.ndarray,
                    known: pl.DataFrame, per_user: int = 40
                    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Return unseen candidates and scores for all previously bought items."""
        if per_user < 1:
            raise ValueError("per_user must be positive")
        users = np.sort(np.unique(users)).astype(np.int64)
        profiles = _profiles(bundle, users, self.product_ids, self.group_size,
                             self.within_decay, self.group_decay)
        train_norm = normalize(self.train_profiles, norm="l2", axis=1)
        query_norm = normalize(profiles, norm="l2", axis=1)
        known_map = dict(known.select("user_id", "product_id").unique()
                         .group_by("user_id").agg("product_id").iter_rows())
        novel_rows, known_rows = [], []
        k = min(self.neighbors, len(self.train_users))
        # Batching bounds the query-to-training similarity matrix; the catalog
        # score for one user is dense, while the stored profiles remain sparse.
        for start in range(0, len(users), 64):
            end = min(start + 64, len(users))
            similarity = (query_norm[start:end] @ train_norm.T).toarray()
            for local, uid in enumerate(users[start:end]):
                uid = int(uid)
                row = similarity[local]
                own_position = np.searchsorted(self.train_users, uid)
                if own_position < len(self.train_users) and self.train_users[own_position] == uid:
                    row[own_position] = 0  # leave-one-user-out for fit users
                if k < len(row):
                    nearest = np.argpartition(row, -k)[-k:]
                else:
                    nearest = np.arange(len(row))
                nearest = nearest[row[nearest] > 0]
                if len(nearest):
                    weight = np.square(row[nearest])
                    weight /= weight.sum()
                    neighbor_score = np.asarray(
                        self.train_profiles[nearest].T @ weight, dtype=np.float32
                    ).ravel()
                else:
                    neighbor_score = np.zeros(len(self.product_ids), dtype=np.float32)
                personal = profiles[start + local].toarray().ravel()
                score = (self.personal_weight * personal +
                         (1 - self.personal_weight) * neighbor_score)
                owned = known_map.get(uid, [])
                for pid in owned:
                    position = np.searchsorted(self.product_ids, pid)
                    if position < len(self.product_ids) and self.product_ids[position] == pid:
                        known_rows.append((uid, int(pid), float(score[position])))
                        score[position] = -np.inf
                positive = np.flatnonzero(score > 0)
                if len(positive) > per_user:
                    top = positive[np.argpartition(score[positive], -per_user)[-per_user:]]
                else:
                    top = positive
                top = sorted(top, key=lambda j: (-score[j], int(self.product_ids[j])))
                novel_rows.extend((uid, int(self.product_ids[j]), float(score[j])) for j in top)
        schema = {"user_id": pl.Int64, "product_id": pl.Int64,
                  "neighbor_score": pl.Float32}
        return (pl.DataFrame(novel_rows, schema=schema, orient="row"),
                pl.DataFrame(known_rows, schema=schema, orient="row"))


def fit_temporal_neighbors(bundle: DataBundle, train_users: np.ndarray,
                           group_size: int = 7, within_decay: float = 0.9,
                           group_decay: float = 0.7, neighbors: int = 50,
                           personal_weight: float = 0.7) -> TemporalNeighborIndex:
    if group_size < 1 or neighbors < 1:
        raise ValueError("group_size and neighbors must be positive")
    if not all(0 < x <= 1 for x in (within_decay, group_decay)):
        raise ValueError("Decay factors must lie in (0, 1]")
    if not 0 <= personal_weight <= 1:
        raise ValueError("personal_weight must lie in [0, 1]")
    train_users = np.sort(np.unique(train_users)).astype(np.int64)
    product_ids = np.sort(bundle.products["product_id"].to_numpy().astype(np.int64))
    profiles = _profiles(bundle, train_users, product_ids, group_size,
                         within_decay, group_decay)
    return TemporalNeighborIndex(product_ids, train_users, profiles, group_size,
                                 within_decay, group_decay, neighbors,
                                 personal_weight)
