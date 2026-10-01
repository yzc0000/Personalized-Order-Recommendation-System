"""An explicitly experimental new-product preview from saved retrievers."""

import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from .data import DataBundle, load_bundle
from .discovery_features import build_discovery_candidates
from .discovery_fusion import fuse_sources, rank_source
from .features import build_features
from .metrics import rank_candidates
from .sparse_discovery import retrieve_popular_index, retrieve_source


def preview_candidates(bundle: DataBundle, index_dir: Path,
                       ranker_dir: Path | None = None,
                       k: int = 5) -> tuple[pl.DataFrame, dict]:
    """Rank unseen products using only completed baskets and saved fit indices."""
    if k < 1:
        raise ValueError("k must be positive")
    report = json.loads((index_dir / "metrics.json").read_text(encoding="utf-8"))
    per_source = int(report["per_source_limit"])
    variant = report["selected_fusion"]
    known = (bundle.history_lines.join(
        bundle.history_orders.select("order_id", "user_id"), on="order_id")
        .select("user_id", "product_id").unique())
    latest = bundle.history_orders.group_by("user_id").agg(
        pl.col("order_number").max().alias("last_order_number")
    )
    last_orders = (bundle.history_orders.join(latest, on="user_id")
                   .filter(pl.col("order_number") == pl.col("last_order_number"))
                   .select("order_id", "user_id"))
    anchors = bundle.history_lines.join(last_orders, on="order_id").select(
        "user_id", "product_id"
    )
    users = np.sort(bundle.history_orders["user_id"].unique().to_numpy())
    popularity_index = joblib.load(index_dir / "popularity_index.joblib")
    neighbor_index = joblib.load(index_dir / "neighbor_index.joblib")
    basket_index = joblib.load(index_dir / "basket_index.joblib")
    neighbor, neighbor_known = neighbor_index.score_users(
        bundle, users, known, per_source,
    )
    popularity_limit = (int(report["fusion_limit"]) if variant in
                        {"behavioral_fill", "popularity_only"} else per_source)
    sources = {
        "neighbor": neighbor,
        "basket": retrieve_source(basket_index, anchors, known, per_source, "basket"),
        "popularity": retrieve_popular_index(
            popularity_index, known, popularity_limit,
        ),
    }
    if variant == "all_four":
        content_index = joblib.load(index_dir / "content_index.joblib")
        sources["content"] = retrieve_source(content_index, anchors, known,
                                              per_source, "content")
    elif variant not in {"behavioral_three", "behavioral_fill", "popularity_only"}:
        raise ValueError(f"Unknown saved retrieval variant: {variant}")
    priority = (["popularity"] if variant == "popularity_only" else
                ["neighbor", "basket", "content", "popularity"])
    fused = fuse_sources([
        rank_source(sources[name], name, priority.index(name))
        for name in priority if name in sources
    ], int(report["fusion_limit"]))
    if not fused.join(known, on=["user_id", "product_id"]).is_empty():
        raise AssertionError("New-product preview contained a previously bought item")
    preview_report = {"ranking": "source_fusion",
                      "pilot_new_only_p5": report.get("test_new_only_fusion_p5")}
    display = (fused.filter(pl.col("rank") <= k)
               .join(bundle.products.select("product_id", "product_name"),
                     on="product_id", how="left", maintain_order="left")
               .sort("user_id", "rank")
               .select("user_id", pl.col("rank").alias("preview_rank"),
                       "product_id", "product_name", "source"))
    ranker_metrics = ranker_dir / "metrics.json" if ranker_dir else None
    if ranker_metrics and ranker_metrics.is_file():
        ranker_report = json.loads(ranker_metrics.read_text(encoding="utf-8"))
        compatible = (ranker_report.get("retrieval_variant") == variant and
                      ranker_report.get("novel_budget") == report["fusion_limit"] and
                      ranker_report.get("train_users") == report["train_users"])
        if compatible:
            if "novel_preview_selected_test" in ranker_report:
                preview_report["pilot_new_only_p5"] = ranker_report[
                    "novel_preview_selected_test"]["precision_at_5"]
            if ranker_report.get("novel_preview_selected") == "specialist":
                model_path = ranker_dir / "novel_preview_model.joblib"
                if not model_path.is_file():
                    raise FileNotFoundError(f"Selected preview model missing: {model_path}")
                targets = pl.DataFrame(schema={"user_id": pl.Int64,
                                               "product_id": pl.Int64})
                _, mixed = build_discovery_candidates(
                    bundle, build_features(bundle), fused, sources,
                    neighbor_known, anchors, targets,
                )
                novel = mixed.filter(pl.col("is_novel") == 1)
                model = joblib.load(model_path)
                display = (rank_candidates(
                    novel.select("user_id", "product_id", "product_name"),
                    model.predict(novel),
                ).filter(pl.col("rank") <= k)
                    .join(fused.select("user_id", "product_id", "source"),
                          on=["user_id", "product_id"], how="left",
                          maintain_order="left")
                    .sort("user_id", "rank")
                    .select("user_id", pl.col("rank").alias("preview_rank"),
                            "product_id", "product_name", "source"))
                preview_report["ranking"] = "specialist_model"
    return display, preview_report


def run_preview(data_dir: Path, index_dir: Path, user_id: int, k: int = 5,
                ranker_dir: Path | None = None) -> None:
    metrics_path = index_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Run python -m experiments.discovery_retrieval first: {metrics_path}")
    variant = json.loads(metrics_path.read_text(encoding="utf-8"))["selected_fusion"]
    required = ["popularity_index.joblib", "neighbor_index.joblib",
                "basket_index.joblib"]
    if variant == "all_four":
        required.append("content_index.joblib")
    missing = [name for name in required if not (index_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Run python -m experiments.discovery_retrieval first; missing {missing}"
        )
    bundle = load_bundle(data_dir, user_id=user_id)
    display, preview_report = preview_candidates(bundle, index_dir, ranker_dir, k)
    score = preview_report.get("pilot_new_only_p5")
    print(f"Experimental new-product preview ({preview_report['ranking']}); "
          "ranking values are not calibrated purchase probabilities.")
    if score is not None:
        print(f"Pilot new-only Precision@5: {score:.3f}")
    if display.is_empty():
        print("No unseen candidate products available.")
    else:
        print(display.drop("user_id"))
