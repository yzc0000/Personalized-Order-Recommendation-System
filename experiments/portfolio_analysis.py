"""Reproducible full-data EDA, score diagnostics and tree explanations."""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.special import expit

from grocery_recommender.data import prepare_data
from grocery_recommender.features import build_features
from grocery_recommender.data import load_bundle
from grocery_recommender.metrics import rank_candidates


BLUE, GOLD, INK, GRID = "#325C9B", "#B9842A", "#263238", "#E3E7EA"


def _style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.edgecolor": INK,
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def _distribution(values: np.ndarray) -> dict:
    return {"count": int(len(values)), "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, .9)),
            "p99": float(np.quantile(values, .99))}


def run_eda(data_dir: Path, output_dir: Path) -> dict:
    prepared = prepare_data(data_dir)
    orders = prepared.orders
    prior_orders = orders.filter(pl.col("eval_set") == "prior")
    labeled_users = orders.filter(pl.col("eval_set") == "train")["user_id"]
    history_count = (prior_orders.filter(pl.col("user_id").is_in(labeled_users.to_numpy()))
                     .group_by("user_id").len()["len"].to_numpy())
    prior_baskets = (pl.scan_csv(data_dir / "order_products__prior.csv")
                     .group_by("order_id").agg(pl.len().alias("items"))
                     .collect(engine="streaming")["items"].to_numpy())
    target_rows = pl.read_csv(data_dir / "order_products__train.csv")
    target_baskets = target_rows.group_by("order_id").len()["len"].to_numpy()
    popularity = prepared.product_stats.sort(
        ["product_purchase_count", "product_id"], descending=[True, False]
    )["product_purchase_count"].to_numpy()
    shares = {}
    for percent in (1, 5, 10):
        cutoff = max(1, int(np.ceil(len(popularity) * percent / 100)))
        shares[f"top_{percent}_percent_products_share"] = float(
            popularity[:cutoff].sum() / popularity.sum()
        )
    report = {
        "population": "all 131,209 labeled next orders and all 3,214,874 prior orders",
        "historical_orders_per_labeled_user": _distribution(history_count),
        "prior_items_per_order": _distribution(prior_baskets),
        "target_items_per_order": _distribution(target_baskets),
        "target_orders_with_fewer_than_five_items": float(
            np.mean(target_baskets < 5)
        ),
        "target_novel_item_share": float(1 - target_rows["reordered"].mean()),
        "product_concentration": shares,
        "first_order_missing_gap_rate": float(
            prior_orders.filter(pl.col("order_number") == 1)
            ["days_since_prior_order"].is_null().mean()
        ),
        "days_since_prior_order_capped_at_30_share_of_nonfirst": float(
            prior_orders.filter(pl.col("order_number") > 1)
            ["days_since_prior_order"].eq(30).mean()
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eda.json").write_text(json.dumps(report, indent=2) + "\n",
                                         encoding="utf-8")
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    axes[0, 0].hist(history_count, bins=np.arange(1.5, 52.5, 1),
                    color=BLUE, edgecolor="white")
    axes[0, 0].set(title="Completed orders per labeled customer",
                   xlabel="Completed orders (2–51 shown)", ylabel="Customers")
    axes[0, 1].hist(prior_baskets, bins=np.arange(.5, 51.5, 1),
                    color=BLUE, edgecolor="white")
    axes[0, 1].set(title="Prior basket sizes", xlabel="Products per order (1–50 shown)",
                   ylabel="Prior orders")
    axes[1, 0].hist(target_baskets, bins=np.arange(.5, 51.5, 1),
                    color=GOLD, edgecolor="white")
    axes[1, 0].set(title="Labeled next-basket sizes",
                   xlabel="Products per order (1–50 shown)", ylabel="Next orders")
    rank = np.arange(1, len(popularity) + 1)
    axes[1, 1].loglog(rank, popularity, color=BLUE, linewidth=1.5)
    axes[1, 1].set(title="Product purchases are concentrated",
                   xlabel="Product rank by prior purchases (log scale)",
                   ylabel="Prior purchases (log scale)")
    for ax in axes.flat:
        ax.grid(axis="y", color=GRID, linewidth=.7)
        ax.set_axisbelow(True)
    fig.suptitle("Instacart purchase history and next baskets", fontsize=15)
    fig.savefig(output_dir / "eda_distributions.png", dpi=180)
    plt.close(fig)
    return report


def run_explanations(data_dir: Path, output_dir: Path, model_dir: Path) -> dict:
    model = joblib.load(model_dir / "model.joblib")
    if model.name != "xgboost":
        raise ValueError("Tree contribution report requires a saved XGBoost model")
    user_ids = np.loadtxt(model_dir / "test_user_ids.csv", dtype=np.int64)
    chosen = np.random.default_rng(42).choice(
        user_ids, size=min(120, len(user_ids)), replace=False,
    )
    prepared = prepare_data(data_dir)
    bundle = load_bundle(data_dir, user_ids=chosen.tolist(), prepared=prepared)
    features = build_features(bundle).candidates.sort("user_id", "product_id")
    scores = model.predict(features)
    ranking = rank_candidates(
        features.select("user_id", "product_id", "product_name", "label"), scores,
    )
    # Explain a reproducible mix of candidate rows rather than just winners.
    indexes = np.random.default_rng(7).choice(
        features.height, size=min(3000, features.height), replace=False,
    )
    selected = features[indexes]
    matrix = model.matrix(selected)
    import xgboost as xgb
    booster = model.estimator.get_booster()
    dmatrix = xgb.DMatrix(matrix)
    contributions = booster.predict(dmatrix, pred_contribs=True)
    margins = booster.predict(dmatrix, output_margin=True)
    if not np.allclose(contributions.sum(axis=1), margins, atol=1e-4):
        raise AssertionError("Tree contributions do not sum to raw model margins")
    if not np.allclose(expit(margins), model.predict(selected), atol=1e-5):
        raise AssertionError("Raw margins do not match saved probabilities")
    importance = np.abs(contributions[:, :-1]).mean(axis=0)
    order = np.argsort(-importance)
    ranked_features = [{"feature": model.feature_columns[int(i)],
                        "mean_absolute_log_odds_contribution": float(importance[i])}
                       for i in order]
    example_ranks = ranking.filter(
        pl.col("user_id").is_in(chosen[:5]) & (pl.col("rank") <= 5)
    ).sort("user_id", "rank")
    example_frame = example_ranks.join(
        features, on=["user_id", "product_id"], how="left", maintain_order="left",
        suffix="_feature",
    )
    example_contributions = booster.predict(
        xgb.DMatrix(model.matrix(example_frame)), pred_contribs=True,
    )
    if not np.allclose(expit(example_contributions.sum(axis=1)),
                       example_ranks["score"].to_numpy(), atol=1e-5):
        raise AssertionError("Example contributions do not recover ranked scores")
    examples = []
    for row, contribution in zip(example_ranks.to_dicts(), example_contributions):
        strongest = np.argsort(-np.abs(contribution[:-1]))[:3]
        examples.append({
            "user_id": row["user_id"], "product_id": row["product_id"],
            "product_name": row["product_name"], "rank": row["rank"],
            "estimated_probability": row["score"],
            "appeared_in_next_basket": bool(row["label"]),
            "baseline_log_odds": float(contribution[-1]),
            "strongest_log_odds_contributions": [
                {"feature": model.feature_columns[int(i)], "value": float(contribution[i])}
                for i in strongest
            ],
        })
    result = {
        "population": "120 sampled users from the 10k standard benchmark test split",
        "explained_candidate_rows": len(indexes),
        "output_scale": "log odds (XGBoost raw margin); expit gives probability",
        "additivity_verified": True,
        "feature_importance": ranked_features,
        "example_top_five": examples,
        "interpretation": "Attributions explain fitted predictions, not causes of purchases.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_explanations.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    _style()
    top = ranked_features[:12][::-1]
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    ax.barh([x["feature"] for x in top],
            [x["mean_absolute_log_odds_contribution"] for x in top],
            color=BLUE)
    ax.set(title="Which features move the saved model's predictions?",
           xlabel="Mean absolute SHAP contribution to log odds")
    ax.grid(axis="x", color=GRID, linewidth=.7)
    ax.set_axisbelow(True)
    fig.savefig(output_dir / "model_explanations.png", dpi=180)
    plt.close(fig)
    return result


def run_calibration(output_dir: Path, full_metrics: Path) -> None:
    data = json.loads(full_metrics.read_text(encoding="utf-8"))
    bins = data["test_probability"]["calibration_bins"]
    _style()
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    ax.plot([0, 1], [0, 1], color=INK, linestyle="--", linewidth=1,
            label="Ideal calibration")
    ax.scatter([row["mean_score"] for row in bins],
               [row["observed_rate"] for row in bins],
               s=[30 + 180 * row["count"] / max(r["count"] for r in bins)
                  for row in bins], color=BLUE, label="Observed score bins")
    ax.set(xlim=(0, 1), ylim=(0, 1),
           title="Reorder probabilities on original test candidates",
           xlabel="Mean predicted probability", ylabel="Observed reorder rate")
    ax.legend(frameon=False)
    ax.grid(color=GRID, linewidth=.7)
    ax.set_axisbelow(True)
    fig.savefig(output_dir / "score_calibration.png", dpi=180)
    plt.close(fig)


def run_display_tradeoff(output_dir: Path, diagnostics_path: Path) -> None:
    """Plot the observed validation cost of showing fewer suggestions."""
    data = json.loads(diagnostics_path.read_text(encoding="utf-8"))["overall"]
    rows = [
        ("Fixed top 5", 5.0, data["model_precision_at_5"], 0.0),
        ("Score >= 0.5", data["cutoff_0_5"]["mean_displayed_per_user"],
         data["cutoff_0_5"]["precision_among_displayed"],
         data["cutoff_0_5"]["fraction_with_no_displayed_items"]),
        ("Score >= 0.6", data["cutoff_0_6"]["mean_displayed_per_user"],
         data["cutoff_0_6"]["precision_among_displayed"],
         data["cutoff_0_6"]["fraction_with_no_displayed_items"]),
    ]
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    colors = [BLUE, GOLD, "#689B80"]
    for (label, shown, precision, empty), color in zip(rows, colors):
        axes[0].scatter(shown, precision, s=90, color=color)
        axes[0].annotate(label, (shown, precision), xytext=(5, 5),
                         textcoords="offset points")
        axes[1].bar(label, shown * precision, color=color)
    axes[0].set(xlim=(0, 5.8), ylim=(0, .8),
                title="Accuracy rises as fewer items are shown",
                xlabel="Mean displayed products per customer",
                ylabel="Precision among displayed products")
    axes[1].set(title="Matched products per customer",
                ylabel="Mean products shown and then bought")
    axes[1].tick_params(axis="x", rotation=15)
    for ax in axes:
        ax.grid(axis="y", color=GRID, linewidth=.7)
        ax.set_axisbelow(True)
    fig.suptitle("Validation display policies: fixed top five versus score cutoffs")
    fig.savefig(output_dir / "display_tradeoff.png", dpi=180)
    plt.close(fig)


def run_top_five_calibration(output_dir: Path, recommendations_path: Path) -> dict:
    """Audit probability calibration on products actually shown in the pilot."""
    rows = pl.read_csv(recommendations_path)
    if rows.height == 0 or not {"score", "label", "rank"}.issubset(rows.columns):
        raise ValueError("Expected saved top-five recommendations with scores and labels")
    scores = rows["score"].to_numpy()
    labels = rows["label"].to_numpy()
    bins = np.minimum(np.floor(scores * 10).astype(int), 9)
    summary = []
    for index in range(10):
        selected = bins == index
        if selected.any():
            summary.append({"lower": index / 10, "upper": (index + 1) / 10,
                            "count": int(selected.sum()),
                            "mean_score": float(scores[selected].mean()),
                            "observed_rate": float(labels[selected].mean())})
    result = {"population": "selected top-five products on 1500 pilot test customers",
              "displayed_products": rows.height, "score_bins": summary}
    (output_dir / "top_five_calibration.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    _style()
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    ax.plot([0, 1], [0, 1], color=INK, linestyle="--", label="Ideal calibration")
    ax.scatter([row["mean_score"] for row in summary],
               [row["observed_rate"] for row in summary],
               s=[35 + 200 * row["count"] / max(x["count"] for x in summary)
                  for row in summary], color=BLUE, label="Top-five score bins")
    ax.set(xlim=(0, 1), ylim=(0, 1),
           title="Probability calibration among shown pilot products",
           xlabel="Mean predicted probability", ylabel="Observed purchase rate")
    ax.grid(color=GRID, linewidth=.7)
    ax.legend(frameon=False)
    ax.set_axisbelow(True)
    fig.savefig(output_dir / "top_five_calibration.png", dpi=180)
    plt.close(fig)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--model-dir", type=Path,
                        default=Path("artifacts/benchmark_10k_standard"))
    parser.add_argument("--full-metrics", type=Path,
                        default=Path("artifacts/full_v2/metrics.json"))
    parser.add_argument("--validation-diagnostics", type=Path,
                        default=Path("artifacts/full_v2/validation_diagnostics.json"))
    args = parser.parse_args()
    eda = run_eda(args.data_dir, args.output_dir)
    explanations = run_explanations(args.data_dir, args.output_dir, args.model_dir)
    run_calibration(args.output_dir, args.full_metrics)
    run_display_tradeoff(args.output_dir, args.validation_diagnostics)
    run_top_five_calibration(args.output_dir,
                             args.model_dir / "test_recommendations.csv")
    print(json.dumps({"eda": eda, "top_explanations": explanations["feature_importance"][:5]},
                     indent=2))
