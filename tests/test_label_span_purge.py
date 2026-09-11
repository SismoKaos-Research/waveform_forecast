"""Purging a label that does not look the same distance ahead for every hour.

A fixed embargo assumes one lookahead. "Days to next event" has a different one
every hour -- two days inside an aftershock sequence, 183 at the longest gap in
this catalogue -- so the worst case is the only safe constant, and embargoing
183 days of a two-year archive to protect the handful of hours that need it
throws most of the record away.

The exact purge drops each hour individually: a training hour survives only if
the event settling its label happened before the block it is validated against
begins. Two ways this goes quietly wrong, and both are here:

* **Off by one at the boundary.** An event falling INSIDE the first hour of val
  needs that hour. Resolving to `p` instead of `p+1` keeps the contaminated
  training hour, and nothing about the result looks wrong.
* **Censoring changes when a label settles.** With a cap, "at least 30 days" is
  knowable 30 days later without waiting for an event that may never arrive, so
  a capped hour settles EARLIER than its next event. Ignoring that purges hours
  that were never contaminated.
"""
import numpy as np
import pandas as pd
import pytest

from waveform_forecast.catalog import label_resolution_index
from waveform_forecast.splits import purge_by_label_span


def hours(n, start="2024-05-01"):
    return pd.date_range(start, periods=n, freq="h")


def events(*stamps):
    return np.sort(np.array([np.datetime64(s) for s in stamps]))


# --- when a label settles --------------------------------------------------

def test_a_label_settles_at_the_hour_holding_its_next_event():
    idx = hours(10)
    ev = events("2024-05-01T05:30:00")
    r = label_resolution_index(idx, ev, feature_hours=0)
    assert r[0] == 6, "the event falls inside hour 5, so hours 0..5 are needed"


def test_an_event_inside_the_boundary_hour_needs_that_hour():
    """The off-by-one. Resolving to the boundary rather than past it keeps a
    training hour whose label was decided inside the validation block."""
    idx = hours(10)
    r = label_resolution_index(idx, events("2024-05-01T05:00:00"), feature_hours=0)
    assert r[0] == 6
    tr, va, te = purge_by_label_span(np.arange(0, 5), np.arange(5, 8),
                                     np.arange(8, 10), r, quiet=True)
    assert 0 not in tr, "an hour settled by an event inside val survived the purge"


def test_an_event_before_the_boundary_is_kept():
    idx = hours(10)
    r = label_resolution_index(idx, events("2024-05-01T03:30:00"), feature_hours=0)
    tr, _, _ = purge_by_label_span(np.arange(0, 5), np.arange(5, 8),
                                   np.arange(8, 10), r, quiet=True)
    assert 0 in tr


def test_a_label_that_never_settles_cannot_be_in_any_split():
    idx = hours(10)
    r = label_resolution_index(idx, events("2024-05-01T02:00:00"), feature_hours=0)
    assert r[5] == len(idx), "no event after hour 5, so its label never settles"
    tr, _, _ = purge_by_label_span(np.arange(0, 8), np.arange(8, 9),
                                   np.arange(9, 10), r, quiet=True)
    assert 5 not in tr


def test_a_cap_lets_a_label_settle_before_its_next_event():
    """"At least 30 days" is knowable 30 days later, without waiting for an
    event that may never come. Ignoring this purges hours that were never at
    risk."""
    idx = hours(200)
    far = events("2024-06-01T00:00:00")            # ~31 days out
    uncapped = label_resolution_index(idx, far, feature_hours=0)
    capped = label_resolution_index(idx, far, cap_days=2, feature_hours=0)
    assert capped[0] < uncapped[0]
    assert capped[0] == 49, "48 hours later, plus the boundary convention"


def test_a_cap_settles_an_hour_with_no_next_event_at_all():
    idx = hours(200)
    r = label_resolution_index(idx, events("2024-05-01T01:00:00"),
                               cap_days=1, feature_hours=0)
    assert r[100] < len(idx), "the cap settles it even with no event ahead"


def test_the_clock_starts_when_the_features_end():
    idx = hours(10)
    ev = events("2024-05-01T02:30:00")
    early = label_resolution_index(idx, ev, feature_hours=0)
    late = label_resolution_index(idx, ev, feature_hours=1.0)
    assert early[2] == 3
    assert late[2] == len(idx), "hour 2's features already cover the event"


def test_a_gapped_hour_grid_is_handled():
    """`align` produces a union index with holes; positions are not clock time."""
    idx = pd.DatetimeIndex(list(hours(3)) + list(hours(3, "2024-06-01")))
    r = label_resolution_index(idx, events("2024-06-01T01:30:00"), feature_hours=0)
    assert r[0] == 5, "three early hours plus two of June before the event"


# --- the purge itself ------------------------------------------------------

def test_val_is_purged_against_test_and_test_is_never_purged():
    """Test labels resolving after the record ends is a property of the record,
    and hours that never resolve were dropped upstream."""
    idx = hours(12)
    r = label_resolution_index(idx, events("2024-05-01T09:00:00"), feature_hours=0)
    tr, va, te = purge_by_label_span(np.arange(0, 4), np.arange(4, 8),
                                     np.arange(8, 12), r, quiet=True)
    assert len(te) == 4, "test was purged"
    assert len(va) == 0, "every val hour settles inside test"


def test_the_cost_follows_the_boundary_gap_not_the_longest_one():
    """The reason for doing it this way at all.

    A fixed embargo has to size itself to the longest gap ANYWHERE in the
    catalogue, because it cannot know where the boundary will fall. The per-hour
    purge only pays for the gap that actually straddles the boundary. Here the
    316-day gap sits well inside train and events are weekly around the split,
    so the purge should cost about a week, not about ten months.
    """
    idx = hours(24 * 400)
    long_gap = ["2024-05-10T00:00:00", "2025-03-22T00:00:00"]      # ~316 days
    weekly = [str(pd.Timestamp("2025-03-22") + pd.Timedelta(days=7 * k))
              for k in range(1, 25)]
    r = label_resolution_index(idx, events(*long_gap, *weekly), feature_hours=0)
    n = len(idx)
    tr_i, va_i, te_i = np.arange(0, 8000), np.arange(8000, 8400), np.arange(8400, n)
    tr, _, _ = purge_by_label_span(tr_i, va_i, te_i, r, quiet=True)
    dropped = len(tr_i) - len(tr)
    assert dropped <= 7 * 24 + 1, f"purged {dropped} h; the boundary gap is a week"
    assert dropped > 0, "the hours just before val must still be purged"


def test_an_empty_val_block_purges_train_against_test_instead():
    idx = hours(12)
    r = label_resolution_index(idx, events("2024-05-01T09:00:00"), feature_hours=0)
    tr, _, _ = purge_by_label_span(np.arange(0, 8), np.array([], dtype=int),
                                   np.arange(8, 12), r, quiet=True)
    assert len(tr) == 0, "every train hour settles inside test"


def test_the_purge_reports_what_it_dropped(capsys):
    idx = hours(12)
    r = label_resolution_index(idx, events("2024-05-01T09:00:00"), feature_hours=0)
    purge_by_label_span(np.arange(0, 4), np.arange(4, 8), np.arange(8, 12), r)
    assert "purged by label span" in capsys.readouterr().out
