"""Variable-length next-order reorder decisions and score diagnostics."""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss


def decision_metrics(labels: np.ndarray, scores: np.ndarray, user_ids: np.ndarray,
                     threshold: float, beta: float = 0.5) -> dict[str, float | int]:
    """Evaluate a yes/no decision for every previously purchased user-product pair."""
    if not (len(labels) == len(scores) == len(user_ids)) or len(labels) == 0:
        raise ValueError("Labels, scores, and users must be nonempty and aligned")
    selected = scores >= threshold
    positives = labels.astype(bool)
    hits = int(np.count_nonzero(selected & positives))
    predicted = int(np.count_nonzero(selected))
    actual = int(np.count_nonzero(positives))
    precision = hits / predicted if predicted else 0.0
    recall = hits / actual if actual else 0.0
    beta_sq = beta * beta
    f_beta = ((1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
              if precision + recall else 0.0)
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    _, inverse = np.unique(user_ids, return_inverse=True)
    predictions_per_user = np.bincount(inverse, weights=selected.astype(np.int8))
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall_of_reorders": float(recall),
        "f0_5": float(f_beta),
        "f1": float(f1),
        "correct_reorders": hits,
        "predicted_reorders": predicted,
        "actual_reorders": actual,
        "mean_predictions_per_user": float(predictions_per_user.mean()),
        "users_with_no_predictions": int(np.count_nonzero(predictions_per_user == 0)),
        "users": len(predictions_per_user),
    }


def choose_threshold(labels: np.ndarray, scores: np.ndarray,
                     user_ids: np.ndarray) -> tuple[float, list[dict]]:
    """Choose a precision-weighted cutoff on separate decision-validation users."""
    if np.unique(labels).size != 2:
        raise ValueError("Both label classes are required to choose a cutoff")
    trials = [
        decision_metrics(labels, scores, user_ids, round(float(value), 2))
        for value in np.arange(0.05, 0.951, 0.01)
    ]
    best = max(trials, key=lambda row: (row["f0_5"], row["precision"], row["threshold"]))
    return float(best["threshold"]), trials


def probability_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    """Measure probability quality without choosing a threshold."""
    if len(labels) != len(scores) or np.unique(labels).size != 2:
        raise ValueError("Aligned labels with both classes are required")
    labels = labels.astype(np.int8)
    scores = scores.astype(float)
    buckets = np.minimum((scores * 10).astype(np.int8), 9)
    bins = []
    for index in range(10):
        mask = buckets == index
        bins.append({
            "lower": index / 10,
            "upper": (index + 1) / 10,
            "count": int(np.count_nonzero(mask)),
            "mean_score": float(scores[mask].mean()) if mask.any() else None,
            "observed_rate": float(labels[mask].mean()) if mask.any() else None,
        })
    prevalence = float(labels.mean())
    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)),
        "constant_prevalence_brier": float(prevalence * (1 - prevalence)),
        "log_loss": float(log_loss(labels, scores)),
        "positive_rate": prevalence,
        "calibration_bins": bins,
    }
