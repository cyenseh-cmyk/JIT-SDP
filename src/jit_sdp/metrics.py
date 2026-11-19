"""Explicit undefined-cohort policy and validation-only thresholds."""
import math


def classification(y, p, threshold=0.5):
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if len(y) != len(p) or not len(y) or not set(y.tolist()) <= {0, 1}:
        raise ValueError("Expected nonempty paired binary labels and probabilities")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid probabilities")
    prediction = p >= threshold
    tp = int(((y == 1) & prediction).sum())
    tn = int(((y == 0) & ~prediction).sum())
    fp = int(((y == 0) & prediction).sum())
    fn = int(((y == 1) & ~prediction).sum())
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    single = len(set(y.tolist())) < 2
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"n": len(y), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "mcc": None if single else ((tp * tn - fp * fn) / denom if denom else 0.0),
            "ap": None if single else float(average_precision_score(y, p)),
            "roc_auc": None if single else float(roc_auc_score(y, p)),
            "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "single_class": single, "constant_prediction": bool(prediction.all() or (~prediction).all()),
            "no_predicted_positives": tp + fp == 0}


def select_threshold(y, p, step=0.01):
    candidates = [round(i * step, 10) for i in range(1, math.ceil(1 / step)) if i * step < 1]
    values = [(classification(y, p, t)["mcc"], t) for t in candidates]
    if any(mcc is None for mcc, _ in values):
        raise ValueError("Threshold selection requires both validation classes")
    return min(values, key=lambda pair: (-pair[0], abs(pair[1] - 0.5), pair[1]))[1]


def sensitivity(rows):
    import numpy as np
    if not rows:
        return {"n": 0, "delta_mcc": None, "mean_abs_change": None, "flip_rate": None}
    y = [r["y_original"] for r in rows]
    original = [r["p_original"] for r in rows]
    changed = [r["p_transformed"] for r in rows]
    # A cohort is one model/seed, so its frozen threshold is shared.
    threshold = rows[0]["threshold"]
    before = classification(y, original, threshold)["mcc"]
    after = classification(y, changed, threshold)["mcc"]
    delta = np.asarray(changed) - np.asarray(original)
    return {"n": len(rows), "delta_mcc": before - after if before is not None and after is not None else None,
            "mean_abs_change": float(np.abs(delta).mean()), "median_abs_change": float(np.median(np.abs(delta))),
            "mean_signed_change": float(delta.mean()),
            "flip_rate": sum(r["yhat_original"] != r["yhat_transformed"] for r in rows) / len(rows)}
