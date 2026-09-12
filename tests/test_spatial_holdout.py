"""Training on one fault zone and testing on another.

Two things have to be true for that to be a measurement rather than a mistake,
and neither is automatic:

**The held-out zone must be standardizable.** `fit_stats` fits mu/sd per
(station, feature) on the training windows. A held-out ZONE appears in no
training window at all, so its statistics would be all-NaN and fall back to the
identity -- the test stations would reach the model raw while the training
stations arrived z-scored. The run would then measure a scaling mismatch and
call it failed transfer. The fix is to fit every station over the training TIME
window, which the Marmara pair was recording throughout even though it was not
the labelled zone.

**The model must not be tied to a station's position.** `MaskedStationPool` is
attention over the station axis and `project` is one Linear shared across
stations, so a model trained on two Aegean stations can be evaluated on two
Marmara ones. That is a property of the architecture rather than a coincidence,
so it is tested as one.
"""
import numpy as np
import pytest
import torch

from waveform_forecast.data import MultiStationWindows, fit_stats
from waveform_forecast.model import MultiStationForecaster


SEQ = 4
N_HOURS = 200


@pytest.fixture
def zoned():
    """Four stations on one grid: 0,1 are the train zone; 2,3 the test zone.

    The test zone is on a deliberately different scale -- site response is a
    property of the station, and that is the whole reason these statistics are
    per-station.
    """
    rng = np.random.default_rng(0)
    x = np.empty((N_HOURS, 4, 3), dtype=np.float32)
    x[:, 0] = rng.normal(0, 1, (N_HOURS, 3))
    x[:, 1] = rng.normal(0, 1, (N_HOURS, 3))
    x[:, 2] = rng.normal(500, 50, (N_HOURS, 3))
    x[:, 3] = rng.normal(500, 50, (N_HOURS, 3))
    present = np.ones((N_HOURS, 4), dtype=bool)
    labels = rng.integers(0, 2, N_HOURS)
    return x, present, labels


TRAIN_Z = np.array([0, 1])
TEST_Z = np.array([2, 3])


# --- standardizing a zone that is in no training window --------------------

def test_a_held_out_zone_is_standardized_not_left_raw(zoned):
    """The bug this exists to prevent: identity stats on the test stations."""
    x, present, _ = zoned
    train_idx = np.arange(SEQ - 1, 120)
    mu, sd = fit_stats(x, present, train_idx, SEQ)
    # The held-out zone's mean is ~500, not the identity's 0.
    assert mu[2].mean() > 100
    assert sd[2].mean() > 1.0
    assert mu[3].mean() > 100


def test_the_old_behaviour_would_have_left_it_at_the_identity(zoned):
    """`all_station_rows=False` reproduces what the spatial run used to get.

    Only meaningful because the held-out stations are absent from the training
    windows, which is what a zone holdout means.
    """
    x, present, _ = zoned
    p = present.copy()
    p[:, 2:] = False                       # zone absent from every train window
    train_idx = np.arange(SEQ - 1, 120)
    mu_old, sd_old = fit_stats(x, p, train_idx, SEQ, all_station_rows=False)
    assert np.allclose(mu_old[2], 0.0) and np.allclose(sd_old[2], 1.0)


def test_statistics_never_reach_past_the_training_block(zoned):
    """The test zone is standardized on its own PAST, not on the test period.

    Fitting a held-out station on its own test-period data would be the
    evaluation split's distribution entering the model. The hours used are
    bounded by the training windows' own extent.
    """
    x, present, _ = zoned
    x = x.copy()
    # Make everything after hour 120 wildly different. If the fit reached into
    # it, the statistics would move.
    x[120:] = 10_000.0
    train_idx = np.arange(SEQ - 1, 120)
    mu, _ = fit_stats(x, present, train_idx, SEQ)
    assert mu.max() < 1_000


def test_per_station_scaling_survives(zoned):
    """Zone statistics are not pooled into one shared number.

    Pooling would make a permanently noisier station look permanently more
    active, which is the reason these are per-station in the first place.
    """
    x, present, _ = zoned
    mu, _ = fit_stats(x, present, np.arange(SEQ - 1, 120), SEQ)
    assert abs(mu[0].mean()) < 10
    assert mu[2].mean() > 100


# --- selecting a zone's slice of the station axis --------------------------

def test_a_dataset_can_use_a_subset_of_the_station_axis(zoned):
    x, present, labels = zoned
    stats = fit_stats(x, present, np.arange(SEQ - 1, 120), SEQ)
    ds = MultiStationWindows(x, present, labels, SEQ, np.arange(SEQ - 1, 120),
                             stats, station_idx=TEST_Z)
    w, m, _ = ds[0]
    assert w.shape == (SEQ, 2, 3)
    assert m.shape == (SEQ, 2)


def test_the_subset_carries_its_own_stations_statistics(zoned):
    """Slicing must take mu/sd along too, or the zone is z-scored by the wrong pair."""
    x, present, labels = zoned
    stats = fit_stats(x, present, np.arange(SEQ - 1, 120), SEQ)
    ds = MultiStationWindows(x, present, labels, SEQ, np.arange(SEQ - 1, 120),
                             stats, station_idx=TEST_Z)
    w, _, _ = ds[0]
    # Correctly standardized, the ~500-scale zone lands near zero, not near 500.
    assert abs(float(w.mean())) < 5.0


def test_selecting_no_subset_is_the_previous_behaviour(zoned):
    x, present, labels = zoned
    stats = fit_stats(x, present, np.arange(SEQ - 1, 120), SEQ)
    idx = np.arange(SEQ - 1, 120)
    a = MultiStationWindows(x, present, labels, SEQ, idx, stats)[0]
    b = MultiStationWindows(x, present, labels, SEQ, idx, stats,
                            station_idx=np.arange(4))[0]
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_the_label_is_unchanged_by_the_station_subset(zoned):
    """Each zone brings its own label array; the slice must not disturb it."""
    x, present, labels = zoned
    stats = fit_stats(x, present, np.arange(SEQ - 1, 120), SEQ)
    ds = MultiStationWindows(x, present, labels, SEQ, np.array([50]), stats,
                             station_idx=TEST_Z)
    assert float(ds[0][2]) == float(labels[50])


# --- the architecture actually transfers -----------------------------------

def test_a_model_trained_on_one_zone_runs_on_another(zoned):
    """Two stations in, two different stations out -- same weights."""
    x, present, labels = zoned
    stats = fit_stats(x, present, np.arange(SEQ - 1, 120), SEQ)
    model = MultiStationForecaster(feat_dim=3, hidden=8).eval()

    def batch(zone):
        ds = MultiStationWindows(x, present, labels, SEQ,
                                 np.arange(SEQ - 1, 20), stats,
                                 station_idx=zone)
        w, m, _ = ds[0]
        return w.unsqueeze(0), m.unsqueeze(0)

    with torch.no_grad():
        a = model(*batch(TRAIN_Z))
        b = model(*batch(TEST_Z))
    assert a.shape == b.shape == (1,)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()


def test_a_zone_of_a_different_width_still_runs(zoned):
    """The pool is over the station axis, so its length is not fixed.

    A zone with three stations and one with two are both valid inputs to the
    same weights -- which is what lets a holdout use whatever the archive has.
    """
    x, present, labels = zoned
    stats = fit_stats(x, present, np.arange(SEQ - 1, 120), SEQ)
    model = MultiStationForecaster(feat_dim=3, hidden=8).eval()
    outs = []
    for zone in (np.array([0]), np.array([0, 1]), np.array([0, 1, 2])):
        ds = MultiStationWindows(x, present, labels, SEQ, np.array([50]),
                                 stats, station_idx=zone)
        w, m, _ = ds[0]
        with torch.no_grad():
            outs.append(model(w.unsqueeze(0), m.unsqueeze(0)))
    assert all(o.shape == (1,) and torch.isfinite(o).all() for o in outs)


def test_station_order_within_a_zone_does_not_change_the_forecast(zoned):
    """Attention pooling is permutation-equivariant over the station axis.

    If it were not, a zone's result would depend on the order the codes were
    typed on the command line.
    """
    x, present, labels = zoned
    stats = fit_stats(x, present, np.arange(SEQ - 1, 120), SEQ)
    model = MultiStationForecaster(feat_dim=3, hidden=8).eval()

    def out(zone):
        ds = MultiStationWindows(x, present, labels, SEQ, np.array([50]),
                                 stats, station_idx=zone)
        w, m, _ = ds[0]
        with torch.no_grad():
            return model(w.unsqueeze(0), m.unsqueeze(0))

    assert torch.allclose(out(np.array([2, 3])), out(np.array([3, 2])),
                          atol=1e-6)
