"""Evaluate the optional new-only preview without changing model selection."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from grocery_recommender.co_basket import retrieve_co_basket_candidates
from grocery_recommender.data import load_bundle, prepare_data
from grocery_recommender.features import FeatureSet, build_features
from grocery_recommender.mixed import build_mixed_candidates
from grocery_recommender.modeling import split_user_ids

from .basket_pilot import _metrics, _targets


def run(data_dir: Path, output_dir: Path) -> dict:
    report = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    prepared = prepare_data(data_dir)
    labeled = prepared.orders.filter(pl.col("eval_set") == "train")["user_id"].to_numpy()
    original_train = split_user_ids(labeled)[0]
    users = np.random.default_rng(42).choice(
        original_train, size=report["sample_users"], replace=False,
    )
    test_users = split_user_ids(users)[2]
    bundle = load_bundle(data_dir, user_ids=test_users.tolist(), prepared=prepared)
    base = build_features(bundle)
    anchors, remaining, sizes = _targets(bundle, report["mode"])
    base = FeatureSet(
        base.candidates.filter(pl.col("user_id").is_in(sizes["user_id"].to_numpy())),
        sizes, remaining.height, 0,
    )
    index = joblib.load(output_dir / "co_basket_index.joblib")
    novel, known_scores = retrieve_co_basket_candidates(
        index, anchors, base.candidates.select("user_id", "product_id"),
        per_user=report["novel_per_user"],
    )
    _, mixed = build_mixed_candidates(
        bundle, base, novel, known_scores, anchors, remaining,
        exclude_anchors=(report["mode"] == "session"),
    )
    model_path = output_dir / (
        "mixed_preview_model.joblib" if report["selected"] != "mixed"
        else "model.joblib"
    )
    model = joblib.load(model_path)
    scores = model.predict(mixed)
    novel_mask = mixed["is_novel"].to_numpy() == 1
    new_only = mixed.filter(pl.col("is_novel") == 1)
    result = {
        "note": "Research preview; no model was selected using these test results",
        "mode": report["mode"],
        "mixed_test": _metrics(mixed, sizes, scores),
        "new_only_test": _metrics(new_only, sizes, scores[novel_mask]),
    }
    (output_dir / "preview_metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/session_pilot"))
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.output_dir), indent=2))
