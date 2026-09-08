"""[ported unchanged from ../forecast, because the label and the floor are
literally the same code -- if these diverge the two projects stop being
comparable, which is the one thing this port exists to prevent.]

Where a forecast leaks: across a block boundary, and into its own window.

Hourly rows are not independent observations. A label looks `horizon_days`
forward and a window looks `seq_hours` back, so two rows on either side of a
fold boundary can be answers to the same earthquake. Two guards:

**The input overlap.** A window ending just after a boundary contains hours from
before it. `seq_hours - 1` of embargo removes that.

**The label overlap, which is the bigger one.** The label at hour H is
determined by events in [H, H + horizon]. Without the extra horizon term, the
last ~14 days of every block carry labels decided by events inside the *next*
block -- train labels encoding what happens in val, and val labels encoding what
happens in test. That is ~9% of samples at horizon 14 d, seq 24 h, and it is the
overlapping-label leakage purging exists to prevent (Lopez de Prado, *Advances
in Financial Machine Learning*, Ch. 7).

`label_hours` has its own version of the same mistake: the horizon must open
where the feature window CLOSES, or the model is shown an event and then asked
to predict it.
"""
import numpy as np
import pandas as pd
import pytest

from waveform_forecast.catalog import (count_events_in_window, days_since_prev_major,
                              label_hours, label_hours_rate_change)
from waveform_forecast.splits import walk_forward_splits


# --- the label -------------------------------------------------------------

def test_the_horizon_opens_where_the_feature_window_closes():
    """An event inside the hour the model is SHOWN must not be its label.

    The features for hour H are aggregated over [H, H+1h], so a horizon opening
    at H counts an event the model can already see and labels it as future.
    """
    idx = pd.date_range("2024-01-01 00:00", periods=3, freq="h")
    inside = np.array([np.datetime64("2024-01-01T00:30")])   # 30 min into hour 0
    assert label_hours(idx, inside, horizon_days=1.0)[0] == 0, \
        "an event inside the feature window is not a forecast"
    assert label_hours(idx, inside, horizon_days=1.0, feature_hours=0)[0] == 1, \
        "feature_hours=0 restores the old behaviour, for reproducing old figures"


def test_the_horizon_is_honoured_to_the_second():
    """timedelta64 with an int day count silently truncated: --horizon-days 0.5
    became a ZERO-day horizon and every label came out negative."""
    idx = pd.date_range("2024-01-01", periods=1, freq="h")
    at_8h = np.array([np.datetime64("2024-01-01T09:00")])
    assert label_hours(idx, at_8h, horizon_days=0.5)[0] == 1
    assert label_hours(idx, at_8h, horizon_days=0.2)[0] == 0


def test_an_event_past_the_horizon_is_not_a_positive():
    idx = pd.date_range("2024-01-01", periods=1, freq="h")
    far = np.array([np.datetime64("2024-02-01T00:00")])
    assert label_hours(idx, far, horizon_days=14.0)[0] == 0


def test_days_since_previous_looks_only_backward():
    idx = pd.date_range("2024-01-10", periods=1, freq="h")
    events = np.array([np.datetime64("2024-01-05"), np.datetime64("2024-01-20")])
    assert days_since_prev_major(idx, events)[0] == pytest.approx(5.0)


def test_days_since_previous_is_nan_before_the_first_event():
    idx = pd.date_range("2024-01-01", periods=1, freq="h")
    assert np.isnan(days_since_prev_major(idx, np.array([np.datetime64("2025-01-01")]))[0])


def test_counting_forward_and_backward_are_not_the_same_window():
    idx = pd.date_range("2024-01-10", periods=1, freq="h")
    ev = np.array([np.datetime64("2024-01-05"), np.datetime64("2024-01-12")])
    assert count_events_in_window(idx, ev, 7.0, forward=True)[0] == 1   # Jan 12
    assert count_events_in_window(idx, ev, 7.0, forward=False)[0] == 1  # Jan 5
    assert count_events_in_window(idx, ev, 1.0, forward=True)[0] == 0   # Jan 12 is 2d out


def test_the_rate_label_is_an_increase_not_a_level():
    """A busy period is not automatically a positive: the target is whether the
    NEXT window holds more than the trailing one."""
    idx = pd.date_range("2024-01-10", periods=1, freq="h")
    ev = np.array([np.datetime64(f"2024-01-{d:02d}") for d in (5, 6, 7, 11)])
    labels, fwd, bwd = label_hours_rate_change(idx, ev, horizon_days=7.0)
    assert bwd[0] == 3 and fwd[0] == 1
    assert labels[0] == 0, "busy behind, quiet ahead -- a DECREASE"


# --- the split -------------------------------------------------------------

def test_folds_are_chronological_and_expanding():
    idx = np.arange(1000)
    folds = walk_forward_splits(idx, n_folds=3)
    prev_train = 0
    for train, val, test in folds:
        assert train.max() < val.min() < test.min(), "a later block must come later"
        assert len(train) > prev_train, "the training window expands"
        prev_train = len(train)


def test_the_embargo_removes_the_rows_that_would_leak():
    """The horizon term is what stops train labels from encoding val's events."""
    idx = np.arange(1000)
    none = walk_forward_splits(idx, n_folds=2, embargo=0)
    purged = walk_forward_splits(idx, n_folds=2, embargo=100)
    for (tr0, va0, _), (tr1, va1, _) in zip(none, purged):
        assert va1.min() - tr1.max() > 100, "a full embargo separates the blocks"
        assert len(va1) < len(va0), "and it costs rows, which is the point"


def test_no_index_appears_in_two_splits_of_one_fold():
    idx = np.arange(500)
    for train, val, test in walk_forward_splits(idx, n_folds=2, embargo=24):
        assert not (set(train) & set(val))
        assert not (set(val) & set(test))
        assert not (set(train) & set(test))


def test_a_later_folds_test_is_never_an_earlier_folds_train():
    """The direction that would be genuine time travel."""
    idx = np.arange(1000)
    folds = walk_forward_splits(idx, n_folds=3, embargo=10)
    for i, (_, _, test_i) in enumerate(folds):
        for j, (train_j, _, _) in enumerate(folds):
            if j < i:
                assert not (set(test_i) & set(train_j))


def test_balanced_folds_place_boundaries_by_positive_mass():
    """Equal hour count lets one sustained swarm fill a block; equal positive
    MASS is decided from the label series before any model runs."""
    idx = np.arange(1000)
    labels = np.zeros(1000, dtype=int)
    labels[:200] = 1                       # all the positives at the front
    plain = walk_forward_splits(idx, n_folds=2)
    massed = walk_forward_splits(idx, n_folds=2, labels=labels)
    assert [len(t) for t, _, _ in plain] != [len(t) for t, _, _ in massed]


def test_an_all_negative_label_series_falls_back_to_equal_width():
    """Rather than dividing by a zero positive mass."""
    idx = np.arange(400)
    labels = np.zeros(400, dtype=int)
    a = walk_forward_splits(idx, n_folds=2, labels=labels)
    b = walk_forward_splits(idx, n_folds=2)
    assert [len(t) for t, _, _ in a] == [len(t) for t, _, _ in b]
