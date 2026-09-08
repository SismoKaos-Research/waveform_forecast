"""Windowing and standardization, where an absent station could become a reading.

`features` writes NaN for an absent station-hour, deliberately, so it cannot be
mistaken for a real average one. That choice pushes two obligations down here:

**The statistics must ignore absent hours.** One NaN in a column makes its mean
NaN, and a NaN mean makes every standardized value NaN. Fitting on present rows
only is what keeps a station with gaps usable at all.

**The tensor handed to the model must be finite.** `NaN * 0` is NaN, so a
masked-out cell left as NaN survives a zero weight, poisons the pooled vector,
the loss, and every gradient after it. The fill to 0 is safe only because
`MaskedStationPool` zeroes masked cells before weighting -- the mask, not the
value, is what carries "absent".

**And a station can be absent for an entire split.** GCAM's archive ends
2024-12, so any later fold has none of it. That is a normal fold, not an error.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from waveform_forecast.data import (MultiStationWindows, fit_stats,
                                    stack, station_feature_columns)

FEATS = ["F1", "F2"]
STATIONS = ["AAA", "BBB"]


def frame(n=200, absent_b=slice(0, 100)):
    """Hourly table shaped like `features` output: NaN where absent."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-05-01", periods=n, freq="h")
    data = {}
    for s, scale in zip(STATIONS, (1.0, 50.0)):     # different site scales
        for f in FEATS:
            data[f"{s}__{f}"] = rng.normal(scale=scale, size=n)
    df = pd.DataFrame(data, index=idx)
    df["present_AAA"] = True
    df["present_BBB"] = True
    df.iloc[absent_b, df.columns.get_indexer([f"BBB__{f}" for f in FEATS])] = np.nan
    df.iloc[absent_b, df.columns.get_loc("present_BBB")] = False
    return df


# --- shaping ---------------------------------------------------------------

def test_the_station_axis_is_stackable_only_in_a_consistent_order():
    cols, feats = station_feature_columns(frame(), STATIONS)
    assert feats == FEATS
    assert cols["AAA"] == ["AAA__F1", "AAA__F2"]


def test_a_missing_station_says_which_ones_exist():
    with pytest.raises(ValueError, match="AAA"):
        station_feature_columns(frame(), ["ZZZ"])


def test_mismatched_feature_sets_are_refused():
    df = frame().drop(columns=["BBB__F2"])
    with pytest.raises(ValueError, match="different feature set"):
        station_feature_columns(df, STATIONS)


def test_stack_gives_hours_stations_features():
    x, present, feats = stack(frame(), STATIONS)
    assert x.shape == (200, 2, 2) and present.shape == (200, 2)
    assert present[:100, 1].sum() == 0 and present[100:, 1].all()


# --- statistics ------------------------------------------------------------

def test_statistics_ignore_absent_hours():
    """One NaN in the column would otherwise make every standardized value NaN."""
    x, present, _ = stack(frame(), STATIONS)
    mu, sd = fit_stats(x, present, np.arange(24, 200), seq_hours=24)
    assert np.isfinite(mu).all() and np.isfinite(sd).all()
    assert (sd > 0).all()


def test_statistics_are_per_station_not_pooled():
    """Site response is the station's, not the earthquake's: one shared mean
    would make a permanently noisier station look permanently more active."""
    x, present, _ = stack(frame(), STATIONS)
    mu, sd = fit_stats(x, present, np.arange(24, 200), seq_hours=24)
    assert sd.shape == (2, 2)
    assert sd[1].mean() > 10 * sd[0].mean(), "the 50x-scale station differs"


def test_a_station_absent_for_the_whole_split_gets_the_identity(capsys):
    """GCAM ends 2024-12; every later fold has none of it. A normal fold."""
    x, present, _ = stack(frame(absent_b=slice(0, 200)), STATIONS)
    mu, sd = fit_stats(x, present, np.arange(24, 200), seq_hours=24)
    assert np.allclose(mu[1], 0.0) and np.allclose(sd[1], 1.0)
    assert np.isfinite(mu).all() and np.isfinite(sd).all()
    assert "absent for the whole training split" in capsys.readouterr().out


# --- what the model receives -----------------------------------------------

def windows(df, indices, seq_hours=24):
    x, present, _ = stack(df, STATIONS)
    labels = np.zeros(len(df), dtype=np.int64)
    stats = fit_stats(x, present, indices, seq_hours)
    return MultiStationWindows(x, present, labels, seq_hours, indices, stats)


def test_the_window_the_model_sees_is_always_finite():
    """NaN * 0 is NaN: a masked cell left as NaN survives its zero weight and
    takes the pooled vector, the loss and every gradient with it."""
    ds = windows(frame(), np.arange(24, 200))
    for i in (0, 50, len(ds) - 1):
        w, m, _ = ds[i]
        assert torch.isfinite(w).all()


def test_absent_cells_are_zero_and_the_mask_says_so():
    ds = windows(frame(), np.arange(24, 200))
    w, m, _ = ds[0]                                # window ends at hour 24
    assert not m[:, 1].any(), "BBB is absent for the first 100 hours"
    assert float(w[:, 1].abs().max()) == 0.0


def test_a_present_cell_is_standardized_not_zeroed():
    ds = windows(frame(), np.arange(24, 200))
    w, m, _ = ds[len(ds) - 1]
    assert m.all(), "both stations present at the end"
    assert float(w.abs().max()) > 0.0
    assert abs(float(w.mean())) < 3.0, "standardized, not raw 50x values"


def test_the_shape_is_time_stations_features():
    ds = windows(frame(), np.arange(24, 200), seq_hours=12)
    w, m, y = ds[0]
    assert w.shape == (12, 2, 2) and m.shape == (12, 2)
    assert y.dtype == torch.float32 and y.shape == ()


def test_the_label_is_the_windows_last_hour():
    x, present, _ = stack(frame(), STATIONS)
    labels = np.arange(200, dtype=np.int64)
    idx = np.arange(24, 200)
    ds = MultiStationWindows(x, present, labels, 24, idx,
                             fit_stats(x, present, idx, 24))
    assert int(ds[0][2]) == 24
    assert int(ds[5][2]) == 29


def test_val_and_test_must_reuse_the_training_statistics():
    """Fitting their own would let the evaluation split's distribution in."""
    x, present, _ = stack(frame(), STATIONS)
    labels = np.zeros(200, dtype=np.int64)
    train_idx = np.arange(24, 120)
    stats = fit_stats(x, present, train_idx, 24)
    test = MultiStationWindows(x, present, labels, 24, np.arange(120, 200), stats)
    assert np.array_equal(test.mu, stats[0]) and np.array_equal(test.sd, stats[1])


def test_a_batch_flows_through_the_model_finite():
    """The end-to-end property: real windows, real mask, finite logits."""
    from torch.utils.data import DataLoader

    from waveform_forecast.model import MultiStationForecaster
    ds = windows(frame(), np.arange(24, 200))
    w, m, _ = next(iter(DataLoader(ds, batch_size=8)))
    torch.manual_seed(0)
    net = MultiStationForecaster(feat_dim=2, hidden=8).eval()
    with torch.no_grad():
        out = net(w, m)
    assert out.shape == (8,) and torch.isfinite(out).all()
