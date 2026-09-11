"""Walk-forward chronological cross-validation and its diagnostics.

Ported unchanged from `cnn_earthquake/src/sismokaos/splits.py`.

Hourly rows are not independent observations: a handful of distinct
earthquakes drive every label in a fold, so the diagnostics print distinct
event counts beside each split rather than row counts alone."""

import warnings

import numpy as np
import pandas as pd


def print_split_diagnostics(hourly_index: pd.DatetimeIndex, labels: np.ndarray,
                            train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray,
                            n_blocks: int = 10, skew_ratio: float = 1.5,
                            quantity: str = "pos rate") -> None:
    """How the label moves over time, in `n_blocks` equal-width windows.

    Args:
        quantity: What the per-block mean is called. The default reads the
            label as a 0/1 rate; a continuous target passes its own name, or
            the block line reports "pos rate 14.000" for a wait measured in
            days and the skew warning talks about a positive rate that does
            not exist.
    """
    n = len(hourly_index)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    split_of = np.full(n, "", dtype=object)
    split_of[train_idx] = "train"
    split_of[val_idx] = "val"
    split_of[test_idx] = "test"

    print(f"\n  {quantity} over time (equal-width blocks):")
    for b in range(n_blocks):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            continue
        block_splits = split_of[lo:hi]
        present = block_splits[block_splits != ""]
        dominant = pd.Series(present).mode().iloc[0] if len(present) else "-"
        # nanmean: a regression target is NaN where the record ends before the
        # next event, and one such hour would blank the whole block. A block
        # that is ENTIRELY past the last event is all-NaN, which nanmean reports
        # as NaN with a RuntimeWarning -- correct, and worth showing as NaN
        # rather than silently omitting, since it says exactly where the label
        # stops being knowable. Silenced here rather than globally so the
        # warning keeps its meaning everywhere else.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            block_mean = np.nanmean(labels[lo:hi])
        print(f"    {hourly_index[lo].date()} .. {hourly_index[hi - 1].date()}  "
               f"{quantity} {block_mean:.3f}  n={hi - lo:4d}  split~{dominant}")

    rates = {name: np.nanmean(labels[idx]) for name, idx in
             (("train", train_idx), ("val", val_idx), ("test", test_idx)) if len(idx)}
    if "train" in rates and "test" in rates and rates["train"] > 0:
        ratio = rates["test"] / rates["train"]
        if ratio > skew_ratio or ratio < 1 / skew_ratio:
            print(f"\n  [!] test {quantity} ({rates['test']:.3f}) is {ratio:.2f}x train's "
                   f"({rates['train']:.3f}) -- likely a swarm or quiet period concentrated in "
                   "one split rather than the model generalizing. Compare against the "
                   "floors below (same skew), not against a nominal baseline.")


def walk_forward_splits(valid_end_indices: np.ndarray, n_folds: int, labels: np.ndarray = None,
                        embargo: int = 0):
    """Expanding-window walk-forward splits."""
    n_blocks = n_folds + 2
    if labels is None:
        edges = np.linspace(0, len(valid_end_indices), n_blocks + 1).astype(int)
    else:
        cum = np.concatenate([[0], np.cumsum(labels)])
        total = cum[-1]
        if total == 0:
            edges = np.linspace(0, len(valid_end_indices), n_blocks + 1).astype(int)
        else:
            targets = np.linspace(0, total, n_blocks + 1)
            edges = np.searchsorted(cum, targets)
            edges[0], edges[-1] = 0, len(valid_end_indices)
            edges = np.maximum.accumulate(edges)
    blocks = [valid_end_indices[edges[i]:edges[i + 1]] for i in range(n_blocks)]
    if embargo > 0:
        for i in range(1, n_blocks):
            if len(blocks[i - 1]) == 0:
                continue
            cutoff = blocks[i - 1][-1] + embargo
            blocks[i] = blocks[i][blocks[i] > cutoff]
    return [(np.concatenate(blocks[:k + 1]), blocks[k + 1], blocks[k + 2])
            for k in range(n_folds)]
