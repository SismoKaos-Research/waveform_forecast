"""Days to the next qualifying event, and the four ways that quietly goes wrong.

The binary label asks "an event within the horizon, yes or no". This asks "how
long until one", and every difference between those two questions is a place to
produce a plausible number that means nothing:

**The record ends.** Every hour after the last qualifying event has no next
event. That is right-censoring, not a long wait, and imputing anything makes the
quietest-looking stretch in the data the one just before the archive stops.

**The lookahead is unbounded.** "Days to next" depends on an event arbitrarily
far ahead, so a training hour near a fold boundary carries a label decided
inside the validation block and no finite embargo removes it. Censoring at the
horizon is what bounds it -- and is what makes the two labels comparable, since
both then depend on events in the same interval.

**The target is heavy-tailed.** A squared error on raw days lets the longest
waits set the gradient, and the model predicts the mean everywhere with a
respectable MAE and no rank correlation at all.

**The floor is not the constant.** Time since the last event is known for free
and is the most informative scalar in the catalogue. A model that cannot beat it
has learnt nothing the clock did not already say.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from waveform_forecast.catalog import days_to_next_major
from waveform_forecast.evaluate import (RegressionFoldResult,
                                        persistence_regression,
                                        regression_fold_result,
                                        summarise_regression)
from waveform_forecast.metrics import regression_report
from waveform_forecast.train import transform_target


def hours(start, n):
    return pd.date_range(start, periods=n, freq="h")


def events(*stamps):
    return np.sort(np.array([np.datetime64(s) for s in stamps]))


# --- the label -------------------------------------------------------------

def test_the_target_is_days_until_the_next_qualifying_event():
    idx = hours("2024-05-01", 49)
    ev = events("2024-05-03T00:00:00")
    days, _ = days_to_next_major(idx, ev, feature_hours=0)
    assert days[0] == pytest.approx(2.0)
    assert days[24] == pytest.approx(1.0)
    assert days[48] == pytest.approx(0.0)


def test_the_clock_starts_when_the_features_end():
    """`hourly_index` holds hour STARTS and hour H's features cover [H, H+1h].
    Measuring from H lets the model read a fraction of its own answer off the
    input -- the same window-end mistake `label_hours` documents."""
    idx = hours("2024-05-01", 1)
    ev = events("2024-05-01T02:00:00")
    from_start, _ = days_to_next_major(idx, ev, feature_hours=0)
    from_end, _ = days_to_next_major(idx, ev, feature_hours=1.0)
    assert from_start[0] == pytest.approx(2 / 24)
    assert from_end[0] == pytest.approx(1 / 24), "one hour of the wait is visible"


def test_hours_past_the_last_event_are_nan_not_a_large_number():
    """The archive ends and the catalogue ends with it. A filled value would
    teach the model that the run-up to the end of the record is quiet."""
    idx = hours("2024-05-01", 96)
    ev = events("2024-05-02T00:00:00")
    days, _ = days_to_next_major(idx, ev, feature_hours=0)
    assert np.isfinite(days[:24]).all()
    assert np.isnan(days[25:]).all()


def test_the_cap_censors_rather_than_dropping_and_says_which():
    """A target of 30.0 because the wait was 30 days and one that is 30.0
    because the wait was nine months are not the same observation."""
    idx = hours("2024-05-01", 3)
    ev = events("2024-09-01T00:00:00")
    days, censored = days_to_next_major(idx, ev, cap_days=14, feature_hours=0)
    assert (days == 14.0).all()
    assert censored.all()


def test_an_uncensored_wait_is_not_marked_censored():
    idx = hours("2024-05-01", 3)
    ev = events("2024-05-03T00:00:00")
    days, censored = days_to_next_major(idx, ev, cap_days=14, feature_hours=0)
    assert not censored.any()
    assert days[0] == pytest.approx(2.0)


def test_the_cap_bounds_the_labels_lookahead():
    """This is what makes the walk-forward purge possible at all: with the cap,
    the label at hour H depends only on events inside [H, H+cap]."""
    idx = hours("2024-05-01", 24)
    near = events("2024-05-10T00:00:00")
    far = events("2024-05-10T00:00:00", "2025-01-01T00:00:00")
    a, _ = days_to_next_major(idx, near, cap_days=14, feature_hours=0)
    b, _ = days_to_next_major(idx, far, cap_days=14, feature_hours=0)
    assert np.array_equal(a, b), "an event beyond the cap changed a label"


# --- the transform ---------------------------------------------------------

def test_log1p_survives_a_zero_wait():
    """The wait is exactly zero at the instant of an event, and log(0) is -inf."""
    out = transform_target(np.array([0.0, 1.0, 14.0]), "log1p")
    assert np.isfinite(out).all()
    assert out[0] == 0.0


def test_the_transform_round_trips_to_days():
    days = np.array([0.0, 0.5, 3.0, 14.0])
    assert np.allclose(np.expm1(transform_target(days, "log1p")), days)


def test_none_leaves_the_target_alone():
    days = np.array([0.0, 3.0, 14.0])
    assert np.array_equal(transform_target(days, "none"), days)


# --- the floors ------------------------------------------------------------

def test_persistence_uses_days_since_previous_and_is_fitted_on_train_only():
    """Fitting it on the split being scored would let that split's distribution
    into the bar the model is measured against."""
    rng = np.random.default_rng(0)
    dsp_tr = rng.uniform(0, 30, 400)
    y_tr = dsp_tr * 0.5 + rng.normal(0, 0.1, 400)      # a monotone relationship
    dsp_te = np.array([1.0, 15.0, 29.0])
    pred = persistence_regression(dsp_tr, y_tr, dsp_te)
    assert pred[0] < pred[1] < pred[2], "the monotone relationship was not learnt"


def test_an_hour_with_no_previous_event_falls_back_to_the_train_median():
    """The opening of the archive. The unconditional median is the only thing
    known about it."""
    dsp_tr = np.linspace(0, 30, 200)
    y_tr = np.linspace(0, 15, 200)
    pred = persistence_regression(dsp_tr, y_tr, np.array([np.nan]))
    assert pred[0] == pytest.approx(np.median(y_tr), abs=0.2)


def test_the_persistence_floor_uses_the_median_not_the_mean():
    """The comparison metric is MAE and the median minimises it. Scoring a
    mean-optimal baseline on an absolute-error metric quietly lowers the bar."""
    dsp_tr = np.zeros(200)
    y_tr = np.concatenate([np.zeros(199), [1000.0]])   # one enormous outlier
    pred = persistence_regression(dsp_tr, y_tr, np.zeros(3))
    assert pred[0] == pytest.approx(0.0), "the mean would be ~5, the median is 0"


def test_the_floor_is_the_better_of_the_two_trivial_rules(capsys):
    """A model has to beat whichever trivial rule wins on that fold, not the
    one that flatters it."""
    rng = np.random.default_rng(1)
    y_tr = rng.uniform(0, 14, 300)
    dsp_tr = y_tr + rng.normal(0, 0.2, 300)
    y_te = rng.uniform(0, 14, 100)
    dsp_te = y_te + rng.normal(0, 0.2, 100)
    r = regression_fold_result("f", y_te, y_te.copy(), [y_te.copy()],
                               y_tr, dsp_tr, dsp_te, quiet=True)
    assert r.floor_mae == min(r.constant_mae, r.persistence_mae)
    assert r.persistence_mae < r.constant_mae, "dsp is informative here"


def test_a_perfect_model_beats_the_floor_and_a_constant_one_does_not():
    rng = np.random.default_rng(2)
    y_tr, y_te = rng.uniform(0, 14, 300), rng.uniform(0, 14, 100)
    dsp_tr, dsp_te = rng.uniform(0, 30, 300), rng.uniform(0, 30, 100)
    good = regression_fold_result("f", y_te, y_te.copy(), [y_te.copy()],
                                  y_tr, dsp_tr, dsp_te, quiet=True)
    flat = np.full(100, float(np.median(y_tr)))
    dull = regression_fold_result("f", y_te, flat, [flat],
                                  y_tr, dsp_tr, dsp_te, quiet=True)
    assert good.beats_floor and good.skill > 0.9
    assert not dull.beats_floor


# --- the collapsed prediction ----------------------------------------------

def test_a_flat_prediction_is_flagged_however_good_its_mae(capsys):
    """The failure mode of a censored target: the model emits the cap
    everywhere. MAE alone looks respectable; the rank correlation is the tell."""
    rng = np.random.default_rng(3)
    y_tr = np.concatenate([np.full(280, 14.0), rng.uniform(0, 14, 20)])
    y_te = np.concatenate([np.full(95, 14.0), rng.uniform(0, 14, 5)])
    flat = np.full(100, 14.0)
    r = regression_fold_result("f", y_te, flat, [flat], y_tr,
                               rng.uniform(0, 30, 300), rng.uniform(0, 30, 100))
    assert np.isnan(r.spearman), "a constant prediction has no rank correlation"
    assert "collapsed to the base rate" in capsys.readouterr().out


def test_the_report_carries_the_prediction_spread():
    r = regression_report(np.arange(50.0), np.full(50, 7.0))
    assert r["pred_std"] == 0.0
    assert np.isnan(r["spearman"])


def test_r2_on_a_constant_target_is_nan_not_zero():
    """Zero would read as "no better than the mean" when the mean is the only
    possible answer."""
    assert np.isnan(regression_report(np.full(20, 3.0), np.full(20, 3.0))["r2"])


# --- the summary -----------------------------------------------------------

def test_the_summary_refuses_a_bare_mae():
    with pytest.raises(TypeError, match="RegressionFoldResult"):
        summarise_regression([{"mae": 1.0}], 1)


def test_the_summary_warns_when_nothing_is_being_ordered(capsys):
    # n=2000, not 100: at 100 samples an independent draw lands at |rho| ~ 0.2
    # often enough that the test would pass or fail on the seed rather than on
    # the behaviour, which is worse than not having it.
    rng = np.random.default_rng(4)
    y_tr, y_te = rng.uniform(0, 14, 2000), rng.uniform(0, 14, 2000)
    dsp = rng.uniform(0, 30, 2000)
    noise = rng.uniform(0, 14, 2000)
    rs = [regression_fold_result(f"f{k}", y_te, noise, [noise], y_tr, dsp,
                                 rng.uniform(0, 30, 2000), quiet=True)
          for k in range(3)]
    summarise_regression(rs, 3)
    assert "not ordering the hours" in capsys.readouterr().out
