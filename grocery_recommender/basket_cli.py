"""Serve the validated basket pilot or explicitly preview new products."""

import json
from pathlib import Path

import joblib
import polars as pl

from .co_basket import retrieve_co_basket_candidates
from .data import load_bundle, validate_bundle
from .features import build_features
from .metrics import rank_candidates
from .mixed import build_mixed_candidates


def basket_recommend(data_dir: Path, output_dir: Path, user_id: int,
                     mode: str, cart_product_ids: list[int] | None = None,
                     preview_new: bool = False, k: int = 5) -> pl.DataFrame:
    """Return recommendations at the same decision point used in the pilot."""
    if k < 1:
        raise ValueError("k must be positive")
    metrics_path = output_dir / "metrics.json"
    index_path = output_dir / "co_basket_index.joblib"
    if not metrics_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(f"Train the {mode} basket pilot first: {output_dir}")
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    if report["mode"] != mode:
        raise ValueError(f"Artifact mode is {report['mode']}, requested {mode}")
    model_path = output_dir / (
        "mixed_preview_model.joblib" if preview_new and report["selected"] != "mixed"
        else "model.joblib"
    )
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing basket model: {model_path}")
    bundle = load_bundle(data_dir, user_id=user_id)
    validate_bundle(bundle)
    if mode == "session":
        if cart_product_ids is None or len(cart_product_ids) != 2 or len(set(cart_product_ids)) != 2:
            raise ValueError("Provide exactly two distinct cart product IDs")
        unknown = set(cart_product_ids) - set(bundle.products["product_id"].to_list())
        if unknown:
            raise ValueError(f"Unknown cart product IDs: {sorted(unknown)}")
        anchors = pl.DataFrame({"user_id": [user_id, user_id],
                                "product_id": cart_product_ids})
    elif mode == "pre_order":
        last_id = bundle.history_orders.sort("order_number")["order_id"][-1]
        anchors = (
            bundle.history_lines.filter(pl.col("order_id") == last_id)
            .select("product_id")
            .with_columns(pl.lit(user_id).alias("user_id"))
            .select("user_id", "product_id")
        )
    else:
        raise ValueError("mode must be session or pre_order")
    base = build_features(bundle)
    index = joblib.load(index_path)
    novel, known_scores = retrieve_co_basket_candidates(
        index, anchors, base.candidates.select("user_id", "product_id"),
        per_user=report["novel_per_user"],
    )
    repeat, mixed = build_mixed_candidates(
        bundle, base, novel, known_scores, anchors,
        exclude_anchors=(mode == "session"),
    )
    if preview_new:
        frame = mixed.filter(pl.col("is_novel") == 1)
    else:
        frame = mixed if report["selected"] == "mixed" else repeat
    if frame.is_empty():
        return pl.DataFrame(schema={"rank": pl.UInt32, "product_id": pl.Int64,
                                    "product_name": pl.String, "score": pl.Float32,
                                    "is_novel": pl.Int8})
    model = joblib.load(model_path)
    return (
        rank_candidates(frame.select("user_id", "product_id", "product_name",
                                     "is_novel"), model.predict(frame))
        .filter(pl.col("rank") <= k)
        .select("rank", "product_id", "product_name", "score", "is_novel")
    )
