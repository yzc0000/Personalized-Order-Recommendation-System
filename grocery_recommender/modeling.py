"""Fit comparable reorder scorers and keep the best ranking model."""

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS
from .metrics import classification_metrics, ranking_metrics


LEGACY_FEATURE_COLUMNS = (
    "user_order_count", "user_item_count", "user_unique_products",
    "user_average_basket_size", "user_mean_days_between_orders",
    "up_purchase_count", "up_reorder_count", "up_purchase_share",
    "up_order_gap", "up_days_since_last_purchase", "up_last_order_fraction",
    "up_mean_cart_position", "product_purchase_count", "product_reorder_rate",
    "user_department_share", "user_aisle_share",
)


def split_user_ids(user_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return reproducible, disjoint 70/15/15 user splits."""
    users = np.sort(np.unique(user_ids))
    if users.size < 10:
        raise ValueError("At least 10 labeled users are needed for a train/validation/test split")
    train, remaining = train_test_split(users, test_size=0.3, random_state=42)
    validation, test = train_test_split(remaining, test_size=0.5, random_state=42)
    return train, validation, test


@dataclass
class Scorer:
    name: str
    estimator: object | None = None
    feature_columns: tuple[str, ...] = tuple(FEATURE_COLUMNS)
    categorical_columns: tuple[str, ...] = ()
    population_stats: pl.DataFrame | None = None

    def matrix(self, frame: pl.DataFrame):
        if self.population_stats is not None:
            additions = [name for name in self.population_stats.columns
                         if name != "product_id" and name not in frame.columns]
            if additions:
                frame = frame.join(
                    self.population_stats.select("product_id", *additions),
                    on="product_id", how="left", maintain_order="left",
                )
        if not self.categorical_columns:
            return frame.select(self.feature_columns).to_numpy().astype(np.float32)
        if self.name != "catboost":
            raise ValueError("Categorical columns currently require CatBoost")
        missing = set(self.categorical_columns) - set(self.feature_columns)
        if missing:
            raise ValueError(f"Categorical columns absent from features: {sorted(missing)}")
        # CatBoost must receive IDs as categories, never their arbitrary
        # numerical ordering. Keep the numerical columns as float32.
        matrix = frame.select(self.feature_columns).to_pandas()
        for name in self.categorical_columns:
            matrix[name] = matrix[name].fillna(-1).astype("int64").astype(str)
        for name in set(self.feature_columns) - set(self.categorical_columns):
            matrix[name] = matrix[name].astype("float32")
        return matrix

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        if self.name == "frequency":
            return frame["up_purchase_share"].to_numpy().astype(np.float32)
        columns = self.__dict__.get("feature_columns", LEGACY_FEATURE_COLUMNS)
        matrix = self.matrix(frame)
        if self.name == "xgboost_ranker":
            return self.estimator.predict(matrix)
        return self.estimator.predict_proba(matrix)[:, 1]


def new_scorer(name: str) -> Scorer:
    if name == "frequency":
        return Scorer(name)
    if name == "logistic":
        return Scorer(name, make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=300, solver="lbfgs"),
        ))
    if name == "random_forest":
        return Scorer(name, RandomForestClassifier(
            n_estimators=100, max_depth=16, min_samples_leaf=20,
            n_jobs=-1, random_state=42,
        ))
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("Install the xgboost extra: pip install -e '.[xgboost]'") from exc
        return Scorer(name, XGBClassifier(
            n_estimators=250, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            eval_metric="logloss", n_jobs=4, random_state=42,
        ))
    if name == "xgboost_ranker":
        try:
            from xgboost import XGBRanker
        except ImportError as exc:
            raise ImportError("Install the xgboost extra: pip install -e '.[xgboost]'") from exc
        return Scorer(name, XGBRanker(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            objective="rank:ndcg", eval_metric="ndcg@10",
            lambdarank_pair_method="topk", lambdarank_num_pair_per_sample=10,
            n_jobs=4, random_state=42,
        ))
    if name == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError("Install the catboost extra: pip install -e '.[catboost]'") from exc
        return Scorer(name, CatBoostClassifier(
            iterations=350, depth=6, learning_rate=0.05,
            verbose=False, thread_count=4, random_seed=42,
            allow_writing_files=False,
        ))
    raise ValueError(f"Unknown model: {name}")


def fit_scorer(scorer: Scorer, train: pl.DataFrame) -> Scorer:
    if scorer.estimator is not None:
        if scorer.name == "xgboost_ranker":
            if "snapshot_id" in train.columns:
                train = train.with_columns(
                    (pl.col("user_id") * 1000 + pl.col("snapshot_id"))
                    .alias("_query_id")
                ).sort("_query_id")
            else:
                train = train.sort("user_id")
        matrix = scorer.matrix(train)
        labels = train["label"].to_numpy()
        if np.unique(labels).size < 2:
            raise ValueError("Both label classes are needed to train a model")
        if scorer.name == "xgboost_ranker":
            query = (train["_query_id"].to_numpy() if "_query_id" in train.columns
                     else train["user_id"].to_numpy())
            scorer.estimator.fit(matrix, labels, qid=query)
        else:
            if scorer.name == "catboost" and scorer.categorical_columns:
                scorer.estimator.fit(matrix, labels,
                                     cat_features=list(scorer.categorical_columns))
            else:
                scorer.estimator.fit(matrix, labels)
    return scorer


def evaluate_scorer(
    scorer: Scorer, validation: pl.DataFrame, target_sizes: pl.DataFrame, k: int
) -> tuple[dict[str, float], np.ndarray]:
    scores = scorer.predict(validation)
    labels = validation["label"].to_numpy()
    if scorer.name == "xgboost_ranker":
        metrics = {"pr_auc": float(average_precision_score(labels, scores))}
    else:
        metrics = classification_metrics(labels, scores)
    metrics.update(ranking_metrics(validation, target_sizes, scores, k))
    return metrics, scores
