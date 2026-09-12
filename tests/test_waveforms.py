"""Turning 86 GB of gapped 100 Hz miniSEED into an hourly tensor.

`--arm raw` and `RawWaveformEncoder` existed from the first commit; the stage
that feeds them did not. `build_inputs` read the hourly FEATURE parquet, so
`--arm raw` handed a 3,072-wide feature vector to a Conv1d expecting three
channels of samples:

    RuntimeError: Given groups=1, weight of size [16, 3, 7], expected
    input[1, 3072, 1] to have 3 channels, but got 3072 channels instead

Three things have to be right for the replacement to be worth running:

**The decimation must filter before it downsamples.** Taking every 20th sample
of a 100 Hz record folds everything above 2.5 Hz into the band that is kept,
and that is exactly where an STA/LTA responds. Aliased noise looks like signal.

**Samples must be placed by absolute time.** The archive is gapped -- one file
holds ~2,000 traces -- so a trace beginning 37 s into an hour must land 37 s in.
Counting from each trace's start puts every sample after the first gap in the
wrong place.

**The amplitudes must survive the write.** Raw counts run past 10^5 and float16
saturates at 65504; a first pass at this silently pinned 52,637 cells of a
single 21-day file to +-inf, which `inf * 0` in the pool turns into NaN.
"""
import numpy as np
import pandas as pd
import pytest

from waveform_forecast.waveforms import (CHANNELS, _factorise, channel_of,
                                         decimate_to)


# --- decimation ------------------------------------------------------------

def test_a_frequency_below_the_new_nyquist_survives():
    t = np.arange(20000) / 100.0
    out = decimate_to(np.sin(2 * np.pi * 1.0 * t), 100, 5)
    assert np.abs(out[100:-100]).max() > 0.9


def test_a_frequency_above_the_new_nyquist_is_removed_not_folded():
    """The whole reason this is not a stride.

    A plain `data[::20]` would alias 30 Hz down into the kept band at full
    amplitude, and it would be indistinguishable from a real 30 Hz-free signal
    with energy there.
    """
    # 32 Hz, not 30: at 5 Hz sampling 30 Hz folds to exactly 0 and would make
    # a plain stride look harmless. 32 Hz folds to 2 Hz -- inside the kept
    # band, at full amplitude, and indistinguishable from real 2 Hz energy.
    t = np.arange(20000) / 100.0
    sig = np.sin(2 * np.pi * 32.0 * t)
    filtered = decimate_to(sig, 100, 5)
    assert np.abs(filtered[100:-100]).max() < 0.05
    strided = sig[::20]
    assert np.abs(strided).max() > 0.9      # what we are NOT doing


def test_the_sample_count_falls_by_the_ratio():
    out = decimate_to(np.zeros(20000), 100, 5)
    assert len(out) == 1000


def test_an_already_correct_rate_is_passed_through():
    data = np.arange(100, dtype=np.float64)
    assert np.allclose(decimate_to(data, 5, 5), data)


def test_the_decimation_factor_is_split_into_stable_stages():
    """scipy's decimate is unstable much above 13x in one pass; 100->5 is 20x."""
    assert _factorise(20) == [10, 2]
    assert all(f <= 10 for f in _factorise(20))
    assert int(np.prod(_factorise(20))) == 20


@pytest.mark.parametrize("n", [2, 4, 5, 8, 10, 20, 40, 100])
def test_the_factors_always_multiply_back(n):
    assert int(np.prod(_factorise(n))) == n


# --- channels --------------------------------------------------------------

def test_the_component_is_the_last_letter_of_the_channel_code():
    assert channel_of("TU.MANT..HHZ") == "Z"
    assert channel_of("TU.DEMI..HHE") == "E"
    assert channel_of("TU.ELBA..HHN") == "N"


def test_other_bands_use_the_same_component_letter():
    """BH?/EH? differ in band and instrument, not in component."""
    assert channel_of("TU.X..BHZ") == "Z"
    assert channel_of("TU.X..EHN") == "N"


def test_the_channel_order_is_fixed():
    """The encoder's first Conv1d takes 3 channels in this order, always."""
    assert CHANNELS == ("E", "N", "Z")


# --- amplitude ------------------------------------------------------------

def test_float16_would_have_clipped_real_counts():
    """Why the tensor is float32. Measured on DEMI: max |count| ~2.6e5.

    This is the regression test for a bug that produced no error at all -- just
    52,637 cells silently pinned to the float16 ceiling.
    """
    counts = np.array([264809.7, -150000.0, 70000.0], dtype=np.float32)
    assert np.isinf(counts.astype(np.float16)).all()
    assert np.isfinite(counts.astype(np.float32)).all()


def test_an_inf_would_not_be_neutralised_by_the_mask():
    """`inf * 0` is NaN, exactly as `NaN * 0` is.

    The mask makes an absent station contribute nothing only because the cells
    are finite first. A saturated amplitude defeats that.
    """
    assert np.isnan(np.float32("inf") * 0.0)


# --- the tensor on disk ----------------------------------------------------

@pytest.fixture
def written(tmp_path):
    """A two-station tensor in the layout `waveforms` writes."""
    hours, n = 12, 18000
    idx = pd.date_range("2024-05-01", periods=hours, freq="h")
    present = {}
    for k, code in enumerate(("AAA", "BBB")):
        arr = np.lib.format.open_memmap(tmp_path / f"{code}.npy", mode="w+",
                                        dtype=np.float32, shape=(hours, 3, n))
        arr[:] = float(k + 1)
        arr.flush()
        del arr
        p = np.ones(hours, dtype=bool)
        p[:k + 1] = False              # AAA misses 1 hour, BBB misses 2
        present[f"present_{code}"] = p
    df = pd.DataFrame(present, index=idx)
    df.index.name = "hour"
    df.to_parquet(tmp_path / "index.parquet")
    return tmp_path


def test_the_tensor_loads_as_hours_stations_channels_samples(written):
    from waveform_forecast.data import load_waveform_tensor
    x, present, idx = load_waveform_tensor(written, ["AAA", "BBB"])
    assert x.shape == (12, 2, 3, 18000)
    assert present.shape == (12, 2)
    assert len(idx) == 12


def test_a_window_stacks_the_stations_without_materialising_the_archive(written):
    """`StackedStations` touches seq_hours rows, not the whole 4.3 GB file."""
    from waveform_forecast.data import load_waveform_tensor
    x, _, _ = load_waveform_tensor(written, ["AAA", "BBB"])
    w = x[2:6]
    assert w.shape == (4, 2, 3, 18000)
    assert np.allclose(w[:, 0], 1.0) and np.allclose(w[:, 1], 2.0)


def test_selecting_a_zone_keeps_only_those_stations(written):
    from waveform_forecast.data import load_waveform_tensor
    x, _, _ = load_waveform_tensor(written, ["AAA", "BBB"])
    sel = x.select([1])
    assert sel.shape == (12, 1, 3, 18000)
    assert np.allclose(sel[0:2], 2.0)


def test_the_mask_comes_from_the_index_not_from_the_samples(written):
    """A gap is written as zeros AND marked absent; the zeros are never data."""
    from waveform_forecast.data import load_waveform_tensor
    _, present, _ = load_waveform_tensor(written, ["AAA", "BBB"])
    assert not present[0, 0] and present[1, 0]
    assert not present[0, 1] and not present[1, 1] and present[2, 1]


def test_a_missing_station_array_is_refused(written):
    from waveform_forecast.data import load_waveform_tensor
    with pytest.raises(SystemExit):
        load_waveform_tensor(written, ["AAA", "NOPE"])


def test_a_missing_directory_says_how_to_build_it(tmp_path):
    from waveform_forecast.data import load_waveform_tensor
    with pytest.raises(SystemExit):
        load_waveform_tensor(tmp_path / "nope", ["AAA"])


# --- statistics over a raw tensor ------------------------------------------

def test_raw_statistics_are_per_channel_not_per_sample(written):
    """18,000 numbers per channel would describe where in the hour a sample sat.

    Pooling over samples is what leaves the amplitude -- the signal -- intact.
    """
    from waveform_forecast.data import fit_stats, load_waveform_tensor
    x, present, _ = load_waveform_tensor(written, ["AAA", "BBB"])
    mu, sd = fit_stats(x, present, np.arange(3, 12), seq_hours=2)
    assert mu.shape == (2, 3, 1)
    assert sd.shape == (2, 3, 1)


def test_raw_statistics_keep_the_stations_apart(written):
    from waveform_forecast.data import fit_stats, load_waveform_tensor
    x, present, _ = load_waveform_tensor(written, ["AAA", "BBB"])
    mu, _ = fit_stats(x, present, np.arange(3, 12), seq_hours=2)
    assert np.allclose(mu[0], 1.0)
    assert np.allclose(mu[1], 2.0)


def test_a_raw_window_reaches_the_model_finite_and_standardized(written):
    """End to end: tensor -> window -> forward pass, one logit out."""
    import torch
    from waveform_forecast.blocks import RawWaveformEncoder
    from waveform_forecast.data import (MultiStationWindows, fit_stats,
                                        load_waveform_tensor)
    from waveform_forecast.model import MultiStationForecaster
    x, present, _ = load_waveform_tensor(written, ["AAA", "BBB"])
    labels = np.zeros(12, dtype=np.int64)
    stats = fit_stats(x, present, np.arange(3, 12), seq_hours=2)
    ds = MultiStationWindows(x, present, labels, 2, np.arange(3, 12), stats)
    w, m, _ = ds[0]
    assert w.shape == (2, 2, 3, 18000)
    assert torch.isfinite(w).all()
    enc = RawWaveformEncoder(out_dim=8)
    model = MultiStationForecaster(feat_dim=8, hidden=8, encoder=enc).eval()
    with torch.no_grad():
        out = model(w.unsqueeze(0), m.unsqueeze(0))
    assert out.shape == (1,) and torch.isfinite(out).all()
