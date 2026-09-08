"""[ported unchanged from ../forecast, because the label and the floor are
literally the same code -- if these diverge the two projects stop being
comparable, which is the one thing this port exists to prevent.]

The floor, and the ways a forecast looks better than it is.

Everything here guards one claim: on this data a model's AUC is not a result.
The positive class is driven by a handful of earthquakes per fold, fold SD runs
0.07-0.16, and the project has already seen a model beat a pooled number while
losing to persistence on most folds. So:

**The floor must be oriented.** An anti-predictive rule is inverted for free, so
the achievable baseline is max(auc, 1-auc). Event mode did not do this, which
collapsed the floor to chance whenever persistence landed below 0.5 -- and made
an n=4 result look like it cleared a 0.5000 bar when the real one was ~0.58.

**A summary must not be assemblable from bare AUCs.** That is the hole the
whole module exists to close, so it is tested as a refusal.
"""
import numpy as np
import pytest

from waveform_forecast.evaluate import (FoldResult, fold_result, persistence_prediction,
                               rate_persistence_auc, summarise)


def make(auc, floor, label="fold 1/2"):
    """A FoldResult built directly, for testing the summary alone."""
    return FoldResult(label=label, auc=auc, floor=floor, base_rate_auc=0.5,
                      persistence_auc=floor, per_seed_aucs=(auc,), n=100,
                      report={})


# --- the floor -------------------------------------------------------------

def test_an_anti_predictive_rule_is_a_floor_not_a_gift():
    """max(auc, 1-auc): a forecaster free to choose the sign of a known
    anti-correlated predictor gets the flipped value for free."""
    y = np.array([0, 0, 1, 1, 0, 1])
    anti = np.array([9.0, 8.0, 1.0, 2.0, 7.0, 0.5])   # high count -> negative
    assert rate_persistence_auc(y, anti) > 0.9


def test_the_floor_is_the_same_whichever_way_the_rule_points():
    y = np.array([0, 0, 1, 1, 0, 1])
    pro = np.array([1.0, 2.0, 8.0, 9.0, 0.5, 7.0])
    anti = -pro
    assert rate_persistence_auc(y, pro) == pytest.approx(rate_persistence_auc(y, anti))


def test_a_single_class_split_gives_no_floor_rather_than_a_wrong_one():
    y = np.zeros(8, dtype=int)
    assert not np.isfinite(rate_persistence_auc(y, np.arange(8.0)))


def test_the_event_floor_is_days_since_previous_within_one_horizon():
    """The trivial rule for `label_hours`: it already happened recently."""
    dsp = np.array([1.0, 30.0, 3.0, 90.0])
    y = np.array([1, 0, 1, 0])
    pred, auc = persistence_prediction("event", y, dsp, horizon_days=14.0)
    assert list(pred) == [1.0, 0.0, 1.0, 0.0]
    assert auc == pytest.approx(1.0)


def test_a_nan_days_since_previous_is_not_a_positive():
    """NaN means no prior event at all -- the start of the record."""
    dsp = np.array([np.nan, 2.0])
    pred, _ = persistence_prediction("event", np.array([0, 1]), dsp, 14.0)
    assert list(pred) == [0.0, 1.0]


# --- the fold result -------------------------------------------------------

def _fold(y, score, dsp, train_labels=None):
    return fold_result("fold 1/1", y, score, [score],
                       np.array([0, 1]) if train_labels is None else train_labels,
                       "event", dsp, 14.0, quiet=True)


def test_a_fold_carries_the_floor_its_auc_was_measured_against():
    y = np.array([0, 0, 1, 1, 0, 1])
    r = _fold(y, np.array([.1, .2, .8, .9, .3, .7]), np.array([30., 40., 1., 2., 50., 3.]))
    assert r.floor >= 0.5
    assert np.isfinite(r.auc)
    assert r.report, "the full metric report comes with it"


def test_the_floor_is_never_below_chance():
    y = np.array([0, 1, 0, 1])
    r = _fold(y, np.array([.4, .6, .3, .7]), np.array([np.nan] * 4))
    assert r.floor >= 0.5


def test_beats_floor_is_strict():
    assert not make(0.60, 0.60).beats_floor
    assert make(0.61, 0.60).beats_floor


def test_a_perfect_persistence_rule_leaves_no_headroom():
    """If the trivial rule already separates the classes, the bar is ~1.0 and a
    model must not be credited for matching it."""
    y = np.array([0, 0, 1, 1])
    dsp = np.array([90., 80., 1., 2.])           # persistence is perfect here
    r = _fold(y, np.array([.1, .2, .8, .9]), dsp)
    assert r.floor == pytest.approx(1.0)
    assert not r.beats_floor


def test_seed_spread_is_reported():
    r = FoldResult("f", 0.6, 0.5, 0.5, 0.5, (0.55, 0.72, 0.61), 10, {})
    assert r.seed_spread == pytest.approx(0.17)


# --- the summary -----------------------------------------------------------

def test_a_bare_auc_cannot_be_summarised():
    """The hole this module exists to close: a number without its floor."""
    with pytest.raises(TypeError, match="carry the floor"):
        summarise([0.63, 0.59], n_folds=2)


def test_the_summary_counts_folds_that_cleared_their_own_floor():
    """Not the mean against the mean -- `feature_lstm` beat its own fold's floor
    in 2 of 5 while losing to persistence on average."""
    out = summarise([make(0.70, 0.55), make(0.52, 0.58)], n_folds=2)
    assert out["folds_beating_floor"] == 1
    assert out["n_folds"] == 2


def test_the_summary_warns_when_spread_swamps_the_margin(capsys):
    """The condition under which a pooled number misleads, said out loud."""
    summarise([make(0.75, 0.55), make(0.45, 0.55)], n_folds=2)
    assert "fold spread exceeds the margin" in capsys.readouterr().out


def test_a_tight_result_gets_no_warning(capsys):
    summarise([make(0.71, 0.55), make(0.70, 0.55)], n_folds=2)
    assert "fold spread exceeds" not in capsys.readouterr().out


def test_no_completed_fold_is_reported_as_such(capsys):
    assert summarise([], n_folds=2) is None
    assert "nothing to summarise" in capsys.readouterr().out
