"""Metric reports, in one console format.

Not a runnable script -- imported only. The subset catalog_mlp uses; bodies are
unchanged from `cnn_earthquake/src/sismokaos/metrics.py`.

`safe_auc`'s `oriented` argument is the one to read carefully: an
anti-predictive rule is exactly as exploitable as a predictive one -- you invert
it -- so a BASELINE's achievable score is max(auc, 1-auc). A trained model gets
no such courtesy: scoring below chance there is a failure to surface, not a sign
to flip.
"""

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

def safe_auc(y, score, oriented=False):
    """Computes ROC-AUC, guarding against an undefined single-class split.

    Args:
        y: True binary labels.
        score: Predicted scores or probabilities for the positive class.
        oriented: If True, return `max(auc, 1 - auc)`. Use this for BASELINES,
            where the sign of the statistic is arbitrary -- a rule scoring 0.20
            is 0.80-accurate once you flip it, so reporting 0.20 as the bar
            understates it. Leave False for a trained model, where scoring
            below chance is a failure to surface, not a sign to flip.

    Returns:
        ROC-AUC as a float, or NaN if `y` contains only one class (AUC is
        undefined in that case).
    """
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    auc = float(roc_auc_score(y, score))
    return max(auc, 1.0 - auc) if oriented else auc

def safe_mcc(y, pred):
    """Computes Matthews correlation coefficient, guarding against degenerate input.

    Args:
        y: True binary labels.
        pred: Predicted binary labels.

    Returns:
        MCC as a float, or NaN if either `y` or `pred` contains only one
        class (MCC is undefined/degenerate in that case).
    """
    y, pred = np.asarray(y), np.asarray(pred)
    if len(np.unique(pred)) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    return float(matthews_corrcoef(y, pred))

def binary_report(y_true, y_score, y_pred=None, threshold=0.5):
    """Full metric set for a binary classifier.

    Accuracy/precision/recall/F1/ROC-AUC/PR-AUC/MCC/Brier/log-loss +
    confusion matrix.

    Args:
        y_true: True binary labels.
        y_score: Predicted positive-class probability.
        y_pred: Predicted binary labels. Defaults to thresholding `y_score`
            at `threshold` when None.
        threshold: Decision threshold used to derive `y_pred` from
            `y_score` when `y_pred` is not given.

    Returns:
        Dict with keys "n", "accuracy", "balanced_accuracy", "precision",
        "recall", "f1", "roc_auc", "pr_auc", "mcc", "brier", "log_loss"
        (each a float, NaN where undefined on a single-class `y_true`), and
        "confusion_matrix" (list of lists).
    """
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    if y_pred is None:
        y_pred = (y_score >= threshold).astype(np.int64)
    y_pred = np.asarray(y_pred)
    single_class = len(np.unique(y_true)) < 2
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_auc(y_true, y_score),
        "pr_auc": float("nan") if single_class else float(average_precision_score(y_true, y_score)),
        "mcc": safe_mcc(y_true, y_pred),
        "brier": float("nan") if single_class else float(brier_score_loss(y_true, y_score)),
        "log_loss": float("nan") if single_class else float(
            log_loss(y_true, np.clip(y_score, 1e-7, 1 - 1e-7))),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

def print_report(name, report, digits=4):
    """Prints one of the `*_report` dicts in a consistent console format.

    Args:
        name: Header printed above the block (e.g. a fold/model label).
        report: A dict returned by `binary_report`, `multiclass_report`, or
            `regression_report`.
        digits: Decimal places used when printing float values.
    """
    print(f"\n--- {name} ---")
    for key, value in report.items():
        if key in ("confusion_matrix", "labels"):
            continue
        if isinstance(value, float):
            print(f"  {key:20s} {value:.{digits}f}")
        else:
            print(f"  {key:20s} {value}")
    if "confusion_matrix" in report:
        print("  confusion_matrix:")
        for row in report["confusion_matrix"]:
            print("   ", row)

def regression_report(y_true, y_pred):
    """Full metric set for a continuous target, in the target's own units.

    `print_report` has documented this function since the port; it had no
    implementation because nothing regressed yet.

    **Spearman is the one to read against the classifier.** MAE depends on the
    censoring cap and on how the wait times happen to be distributed in a fold,
    so it is not comparable across folds, let alone across labels. The rank
    correlation is, and it is the same question ROC-AUC answers for the binary
    label: does the model order the hours correctly. A model with a fine MAE and
    a Spearman near zero has learnt the base rate and nothing else -- which is
    exactly what predicting the training median achieves.

    Args:
        y_true: Observed values.
        y_pred: Predicted values, same units.

    Returns:
        Dict with "n", "mae", "rmse", "median_ae", "r2", "spearman",
        "pearson", and "pred_std" (NaN where undefined).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    if len(y_true) == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"),
                "median_ae": float("nan"), "r2": float("nan"),
                "spearman": float("nan"), "pearson": float("nan"),
                "pred_std": float("nan")}
    err = y_pred - y_true
    var = float(np.var(y_true))
    return {
        "n": len(y_true),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "median_ae": float(np.median(np.abs(err))),
        # NaN, not 0, on a constant target: R^2 is undefined there, and
        # reporting 0 would read as "no better than the mean" when the mean is
        # the only possible answer.
        "r2": float("nan") if var == 0 else float(1.0 - np.mean(err ** 2) / var),
        "spearman": safe_spearman(y_true, y_pred),
        "pearson": safe_pearson(y_true, y_pred),
        # A collapsed prediction is the failure mode of a censored target: the
        # model learns the cap and emits it everywhere. A near-zero spread says
        # so immediately, where MAE alone would look respectable.
        "pred_std": float(np.std(y_pred)),
    }

def safe_spearman(y_true, y_pred):
    """Rank correlation, NaN when either side is constant."""
    from scipy.stats import spearmanr
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) < 3 or len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return float("nan")
    return float(spearmanr(y_true, y_pred).statistic)

def safe_pearson(y_true, y_pred):
    """Linear correlation, NaN when either side is constant."""
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    if len(y_true) < 3 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])
