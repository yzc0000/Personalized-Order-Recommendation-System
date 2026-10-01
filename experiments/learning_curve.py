"""Compare training population sizes on the same full-run held-out users."""

import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from grocery_recommender.data import load_bundle, prepare_data
from grocery_recommender.features import FEATURE_COLUMNS, build_features
from grocery_recommender.metrics import ranking_metrics
from grocery_recommender.modeling import new_scorer, split_user_ids


def _batches(data_dir: Path, prepared, users: np.ndarray, batch_size: int = 5000):
    for start in range(0, len(users), batch_size):
        batch = load_bundle(data_dir, user_ids=users[start:start + batch_size].tolist(),
                            prepared=prepared)
        yield build_features(batch)


def run(data_dir: Path = Path("."), output_dir: Path = Path("artifacts")) -> None:
    prepared = prepare_data(data_dir)
    labeled = prepared.orders.filter(pl.col("eval_set") == "train")["user_id"].to_numpy()
    train_users, validation_users, test_users = split_user_ids(labeled)
    pool = np.random.default_rng(42).permutation(np.concatenate((train_users, validation_users)))
    sizes = (14000, 40000)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="learning_", dir=output_dir) as tmp:
        matrix_path = Path(tmp) / "train.npy"
        label_path = Path(tmp) / "labels.npy"
        # Maximum possible row count is bounded by the prior item count for these users.
        selected = prepared.orders.filter(pl.col("user_id").is_in(pool[:max(sizes)].tolist()))
        prior_ids = selected.filter(pl.col("eval_set") == "prior").select("order_id")
        prior_lines = pl.scan_csv(data_dir / "order_products__prior.csv").join(
            prior_ids.lazy(), on="order_id", how="semi"
        ).select(pl.len()).collect(engine="streaming").item()
        matrix = np.lib.format.open_memmap(matrix_path, mode="w+", dtype="float32",
                                           shape=(prior_lines, len(FEATURE_COLUMNS)))
        labels = np.lib.format.open_memmap(label_path, mode="w+", dtype="int8",
                                           shape=(prior_lines,))
        offsets = {}
        offset = 0
        boundaries = [0, 5000, 10000, 14000, 19000, 24000, 29000, 34000,
                      39000, 40000]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            feature_set = next(_batches(data_dir, prepared, pool[start:end]))
            frame = feature_set.candidates
            count = frame.height
            matrix[offset:offset + count] = frame.select(FEATURE_COLUMNS).to_numpy().astype(np.float32)
            labels[offset:offset + count] = frame["label"].to_numpy()
            offset += count
            users_seen = end
            if users_seen in sizes:
                offsets[users_seen] = offset
            print(f"Training features: {users_seen} users, {offset} candidates", flush=True)
        test_parts = list(_batches(data_dir, prepared, test_users))
        test = pl.concat([part.candidates for part in test_parts])
        test_sizes = pl.concat([part.target_sizes for part in test_parts])
        results = {}
        for size in sizes:
            scorer = new_scorer("xgboost")
            scorer.estimator.set_params(n_estimators=600)
            print(f"Fitting {size}-user comparison model...", flush=True)
            scorer.estimator.fit(matrix[:offsets[size]], labels[:offsets[size]])
            scores = scorer.predict(test)
            results[str(size)] = {
                "candidate_pairs": offsets[size],
                **ranking_metrics(test, test_sizes, scores, 5),
            }
            print(f"{size} users: Precision@5={results[str(size)]['precision_at_5']:.4f}",
                  flush=True)
        full = joblib.load(output_dir / "full" / "evaluated_model.joblib")
        results["full"] = ranking_metrics(test, test_sizes, full.predict(test), 5)
        expected = json.loads((output_dir / "full" / "metrics.json").read_text())
        if abs(results["full"]["precision_at_5"] -
               expected["test_selected_model"]["5"]["precision_at_5"]) > 1e-8:
            raise ValueError("Held-out users or features disagree with the full run")
        data = {
            "test_users": len(test_users),
            "same_test_users": True,
            "boosting_rounds": 600,
            "subset_from": "full train plus validation users, excluding held-out test users",
            "results": results,
        }
        (output_dir / "learning_curve.json").write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        del matrix, labels


if __name__ == "__main__":
    run()
