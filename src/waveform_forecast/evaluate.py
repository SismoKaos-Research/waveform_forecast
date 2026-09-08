"""The floor, the folds, and the one function that reports them together.

Not a runnable script -- imported only, by `train.py`.

**Why this is a module rather than a block inside the trainer.** The central
methodological finding of this project's forecasting work is that a pooled
number misleads on this data:

  - *"Read the block-level numbers, not the pooled ones"*
  - fold SD of 0.07-0.16 dwarfs every gap between models
  - `feature_lstm` "beats its own fold's floor in 2 of 5 folds" while losing to
    persistence on average

A model's AUC alone is therefore not a result. It is a result only beside the
floor it had to clear on *that fold*, and beside the spread across folds. So
`fold_result` computes all three at once and there is no path that produces one
without the others -- `summarise` then refuses to print a summary assembled any
other way.

**The orientation bug this encodes.** An anti-predictive persistence rule is
exactly as exploitable as a predictive one: you invert it. So the achievable
baseline is `max(auc, 1 - auc)`. Rate mode always did this inside
`rate_persistence_auc`; event mode did not, which silently collapsed the floor
to chance whenever persistence landed below 0.5. That is what made an n=4
event-mode result look like it cleared a 0.5000 floor when the properly
oriented bar was ~0.58.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import brier_score_loss

from waveform_forecast.metrics import binary_report, print_report, safe_auc


def rate_persistence_auc(labels: np.ndarray, trailing_counts: np.ndarray) -> float:
    """AUC of the trivial trailing-rate baseline for `label_hours_rate_change`.

    The trailing count is a legitimate backward-looking predictor, but its
    relationship to a rate-INCREASE label is inverted (Omori decay: busy now
    implies calmer next). Reports the achievable baseline as
    `max(auc, 1 - auc)`, since a forecaster free to choose the sign of a
    known-anti-correlated predictor gets the flipped value for free -- so
    that, not 0.5, is the bar a model has to clear.

    Args:
        labels: 0/1 rate-increase labels.
        trailing_counts: Trailing-window event counts, same length.

    Returns:
        Baseline AUC in [0.5, 1.0], or NaN if `labels` is single-class.
    """
    auc = safe_auc(labels, trailing_counts.astype(np.float64))
    return float(max(auc, 1.0 - auc)) if np.isfinite(auc) else float("nan")


@dataclass(frozen=True)
class FoldResult:
    """One fold's model score and the bar it had to clear, together.

    Frozen and constructed only by `fold_result`, so a summary cannot be built
    from an AUC whose floor was never computed.
    """

    label: str
    auc: float
    floor: float
    base_rate_auc: float
    persistence_auc: float
    per_seed_aucs: tuple
    n: int
    report: dict

    @property
    def beats_floor(self):
        return bool(np.isfinite(self.auc) and self.auc > self.floor)

    @property
    def seed_spread(self):
        a = [x for x in self.per_seed_aucs if np.isfinite(x)]
        return float(max(a) - min(a)) if len(a) > 1 else 0.0


def persistence_prediction(label_mode, y_true, dsp_test, horizon_days,
                           rate_trailing_test=None, rate_trailing_train=None):
    """The trivial backward-looking rule for this label, and its AUC.

    Two labels, two different honest floors:

    - `rate`: the trailing count, oriented, via `rate_persistence_auc`.
    - `event` (the default): "a qualifying event happened within one horizon
      already", i.e. `days_since_prev <= horizon_days`.

    (`cnn_earthquake` has a third, `detect`, whose label is itself a threshold
    on dsp -- a dsp rule would reproduce it and score ~1.0, so the base rate was
    its only honest floor. That mode needed the waveform branch and is not in
    this project.)

    Returns:
        (prediction array, auc).
    """
    if rate_trailing_test is not None:
        auc = rate_persistence_auc(y_true, rate_trailing_test)
        thresh = np.median(rate_trailing_train)
        pred = (rate_trailing_test < thresh).astype(np.float64)
        if safe_auc(y_true, rate_trailing_test.astype(np.float64)) > 0.5:
            pred = 1.0 - pred      # trailing rate positively correlated here
        return pred, auc
    pred = np.where(np.isnan(dsp_test), 0,
                    (dsp_test <= horizon_days).astype(int)).astype(np.float64)
    return pred, safe_auc(y_true, pred)


def fold_result(label, y_true, ensemble_score, per_seed_scores, train_labels,
                label_mode, dsp_test, horizon_days, rate_trailing_test=None,
                rate_trailing_train=None, quiet=False, model_name="model"):
    """Scores one fold against its own floor, and prints both.

    This is the only way a `FoldResult` is made. Everything the summary needs
    is computed here, so there is no route to an AUC without the bar beside it.
    """
    y_true = np.asarray(y_true)
    single_class = len(np.unique(y_true)) < 2

    pos_tr = float(np.mean(train_labels))
    base_pred = np.full_like(y_true, int(round(pos_tr)), dtype=np.float64)
    base_auc = safe_auc(y_true, base_pred)

    pers_pred, pers_auc = persistence_prediction(
        label_mode, y_true, dsp_test, horizon_days,
        rate_trailing_test, rate_trailing_train)
    if pers_pred is None:
        pers_pred, pers_auc = base_pred, base_auc
    pers_brier = (float("nan") if single_class
                  else float(brier_score_loss(y_true, pers_pred)))

    # See the module docstring: an anti-predictive rule is inverted for free,
    # so the bar is the oriented value, not the raw one.
    oriented = max(pers_auc, 1.0 - pers_auc) if np.isfinite(pers_auc) else 0.5
    floor = max(0.5, base_auc, oriented)

    per_seed = tuple(safe_auc(y_true, s) for s in per_seed_scores)
    auc = safe_auc(y_true, ensemble_score)
    report = binary_report(y_true, ensemble_score)
    report["brier_skill_score_vs_persistence"] = (
        float("nan") if (single_class or not np.isfinite(pers_brier) or pers_brier == 0)
        else 1.0 - report["brier"] / pers_brier)

    if not quiet:
        print("\n--- Floors (test set) ---")
        print(f"  base-rate (majority)   AUC {base_auc:.4f}   n={len(y_true)}")
        print(f"  persistence            AUC {pers_auc:.4f}   "
              f"Brier {pers_brier:.4f}   n={len(y_true)}")
        print(f"  -> floor to clear      AUC {floor:.4f}   "
              f"(oriented: a rule below 0.5 is inverted for free)")
        print(f"\n--- {model_name} ---")
        print(f"  per-seed AUC: {[f'{a:.4f}' for a in per_seed]}  "
              f"mean {np.mean(per_seed):.4f}  "
              f"spread {max(per_seed) - min(per_seed):.4f}")
        print(f"  ENSEMBLE (mean of {len(per_seed)} seeds' probabilities)   "
              f"AUC {auc:.4f}   n={len(y_true)}")
        print_report(f"{model_name} ensemble ({label}, test set)", report)

    return FoldResult(label=label, auc=float(auc), floor=float(floor),
                      base_rate_auc=float(base_auc),
                      persistence_auc=float(pers_auc),
                      per_seed_aucs=per_seed, n=int(len(y_true)), report=report)


def summarise(results, n_folds):
    """Prints the walk-forward summary: folds, floors, and how often it cleared.

    Raises:
        TypeError: If handed anything but `FoldResult`s. The point of this
            module is that a number cannot be reported without the floor it was
            measured against, and accepting a bare AUC here would be the hole.
    """
    bad = [r for r in results if not isinstance(r, FoldResult)]
    if bad:
        raise TypeError(
            "summarise() takes FoldResult objects, which carry the floor their "
            "AUC was measured against. A bare AUC has no meaning here: fold SD "
            "on this data is 0.07-0.16, and models have beaten the pooled "
            "number while losing to persistence on most folds.")
    if not results:
        print("\n  no fold completed -- nothing to summarise")
        return None

    aucs = np.array([r.auc for r in results], dtype=float)
    floors = np.array([r.floor for r in results], dtype=float)
    beat = sum(r.beats_floor for r in results)
    print(f"\n{'=' * 64}\nWalk-forward CV summary ({len(results)}/{n_folds} folds)\n{'=' * 64}")
    print(f"  ensemble AUC per fold: {[f'{a:.4f}' for a in aucs]}")
    print(f"  ensemble AUC:  mean {np.nanmean(aucs):.4f}  std {np.nanstd(aucs):.4f}")
    print(f"  floor AUC:     mean {np.nanmean(floors):.4f}  std {np.nanstd(floors):.4f}")
    print(f"  beats its own fold's floor in {beat}/{len(results)} folds")
    if len(results) > 1 and np.nanstd(aucs) > abs(np.nanmean(aucs) - np.nanmean(floors)):
        print("  [!] fold spread exceeds the margin over the floor -- read the "
              "per-fold column,\n      not the mean. This is the condition under "
              "which a pooled number misleads.")
    return {"mean_auc": float(np.nanmean(aucs)), "std_auc": float(np.nanstd(aucs)),
            "mean_floor": float(np.nanmean(floors)), "folds_beating_floor": beat,
            "n_folds": len(results)}
