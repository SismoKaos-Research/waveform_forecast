"""Putting several stations on one clock, and the ways that quietly goes wrong.

This module does not compute features -- `sismokaos-cli` and
`Sismokaos-featureExtract` do. It does the part neither does, and every step of
it has a failure that produces a plausible table rather than an error:

**The clock.** `Zaman_Dk` is MINUTES since the epoch. Read as seconds it lands
in 1970, every label joins against nothing, and an empty join is an empty table.

**The aggregation.** ~72 windows an hour collapse to one row. STA/LTA and PEAK
measure transients; their hourly *mean* is the noise floor around the transient,
so losing `max` loses the only thing they measured.

**The presence mask.** An hour with no station is not an hour of quiet. Absent
cells must stay NaN -- a zero is a reading the model cannot distinguish from a
real average one -- and, the case that actually bit: an hour whose every window
failed in the extractor aggregates to all-NaN while still counting 72 rows, so
counting rows rather than usable rows marks it present and hands the model an
empty observation.

**The union.** Stations do not share a span. On the real archives all three
Aegean stations overlap on 80 days of ~700; aligning on the intersection throws
the rest away.
"""
import numpy as np
import pandas as pd
import pytest

from waveform_forecast.features import (DEFAULT_AGGS, align,
                                        read_extractor_output, to_hourly)

FEATURES = ["Z_STA_LTA_Max", "Z_HJORTH_ACTIVITY", "EN_CROSS_CORR"]


def extractor_frame(start, hours, step_s=50, nan_rows=(), seed=0):
    """A stand-in for one extractor's output: a window every `step_s` seconds."""
    rng = np.random.default_rng(seed)
    n = int(hours * 3600 / step_s)
    t0 = pd.Timestamp(start, tz="UTC")
    zaman = [(t0 + pd.Timedelta(seconds=i * step_s)).timestamp() / 60.0 for i in range(n)]
    df = pd.DataFrame({"Pencere_ID": [f"w{i:05d}" for i in range(n)],
                       "Zaman_Dk": zaman,
                       **{c: rng.normal(size=n) for c in FEATURES}})
    for i in nan_rows:
        df.loc[i, FEATURES] = np.nan
    return df


@pytest.fixture
def one(tmp_path):
    p = tmp_path / "a.parquet"
    extractor_frame("2024-05-01", hours=6).to_parquet(p)
    return p


# --- the clock -------------------------------------------------------------

def test_zaman_dk_is_read_as_minutes(one):
    """Read as seconds this lands in 1970 and joins against no label at all."""
    df = read_extractor_output(one)
    assert df.index[0] == pd.Timestamp("2024-05-01", tz="UTC")
    assert df.index.year[0] == 2024


def test_the_id_columns_do_not_reach_the_model(one):
    df = read_extractor_output(one)
    assert "Pencere_ID" not in df.columns and "Zaman_Dk" not in df.columns


def test_a_file_without_zaman_dk_is_refused(tmp_path):
    p = tmp_path / "bad.parquet"
    pd.DataFrame({"Z_STA_LTA_Max": [1.0]}).to_parquet(p)
    with pytest.raises(SystemExit, match="Zaman_Dk"):
        read_extractor_output(p)


def test_csv_and_parquet_agree(tmp_path):
    """The Rust extractor writes parquet, the Python one CSV."""
    raw = extractor_frame("2024-05-01", hours=2)
    a, b = tmp_path / "x.parquet", tmp_path / "x.csv"
    raw.to_parquet(a); raw.to_csv(b, index=False)
    assert np.allclose(read_extractor_output(a).to_numpy(),
                       read_extractor_output(b).to_numpy())


# --- the aggregation -------------------------------------------------------

def test_an_hour_is_one_row(one):
    h = to_hourly(read_extractor_output(one))
    assert len(h) == 6
    assert h.index[0] == pd.Timestamp("2024-05-01", tz="UTC")


def test_max_survives_so_a_transient_does(one):
    """A transient in one window must reach the hourly row. Its mean would be
    the ~72 quiet windows around it."""
    df = read_extractor_output(one)
    df.iloc[10, df.columns.get_loc("Z_STA_LTA_Max")] = 999.0
    h = to_hourly(df)
    assert h["Z_STA_LTA_Max_max"].max() == 999.0
    assert h["Z_STA_LTA_Max_mean"].max() < 100.0, "the mean buries it, as expected"


def test_every_requested_statistic_appears(one):
    h = to_hourly(read_extractor_output(one), aggs=DEFAULT_AGGS)
    for f in FEATURES:
        for a in DEFAULT_AGGS:
            assert f"{f}_{a}" in h.columns


def test_n_windows_counts_usable_rows_not_rows(tmp_path):
    """The bug this pins: the extractor writes an all-NaN row for a window it
    could not compute. Counting rows makes an hour of those look full."""
    p = tmp_path / "n.parquet"
    extractor_frame("2024-05-01", hours=2, nan_rows=range(72)).to_parquet(p)
    h = to_hourly(read_extractor_output(p))
    assert h["n_windows"].iloc[0] == 0, "hour 1 is entirely failed windows"
    assert h["n_windows"].iloc[1] == 72


# --- alignment and presence ------------------------------------------------

def two_stations(tmp_path, nan_rows=()):
    a, b = tmp_path / "A.parquet", tmp_path / "B.parquet"
    extractor_frame("2024-05-01", hours=6, nan_rows=nan_rows, seed=1).to_parquet(a)
    extractor_frame("2024-05-01 03:00", hours=6, seed=2).to_parquet(b)   # 3 h overlap
    return {"AAA": to_hourly(read_extractor_output(a)),
            "BBB": to_hourly(read_extractor_output(b))}


def test_stations_are_aligned_on_the_union_not_the_intersection(tmp_path):
    """On the real archives the intersection is 80 days of ~700."""
    frame, feats, stations = align(two_stations(tmp_path))
    assert stations == ["AAA", "BBB"]
    assert len(frame) == 9, "6 + 6 hours overlapping by 3"
    assert int((frame.present_AAA & frame.present_BBB).sum()) == 3


def test_an_absent_station_is_nan_and_never_zero(tmp_path):
    """Zero after standardization means 'exactly average' -- a reading the model
    cannot tell from a real one."""
    frame, feats, _ = align(two_stations(tmp_path))
    cols = [f"AAA__{c}" for c in feats]
    absent = frame.loc[~frame.present_AAA, cols]
    assert len(absent) == 3
    assert absent.isna().all().all()
    assert not (absent == 0).any().any()


def test_presence_means_the_hour_actually_carries_data(tmp_path):
    """An hour of entirely-failed windows clears any row count and must still
    be absent -- otherwise the model gets an all-NaN row flagged present."""
    frame, feats, _ = align(two_stations(tmp_path, nan_rows=range(72)),
                            min_windows=1)
    assert not bool(frame.present_AAA.iloc[0]), "hour 1 was all failed windows"
    cols = [f"AAA__{c}" for c in feats]
    present_rows = frame.loc[frame.present_AAA, cols]
    assert present_rows.notna().any(axis=1).all()


def test_a_thin_hour_is_absent_rather_than_a_reading(tmp_path):
    """The tail of a chunk is a few windows, not an hour of observation."""
    p = tmp_path / "thin.parquet"
    extractor_frame("2024-05-01", hours=1).head(3).to_parquet(p)
    frame, _, _ = align({"AAA": to_hourly(read_extractor_output(p))}, min_windows=8)
    assert not frame.present_AAA.any()


def test_the_mask_is_boolean_and_complete(tmp_path):
    frame, _, stations = align(two_stations(tmp_path))
    for s in stations:
        assert frame[f"present_{s}"].dtype == bool
        assert frame[f"present_{s}"].notna().all()


def test_columns_are_namespaced_by_station(tmp_path):
    """Two stations carry the same feature names; without the prefix one
    silently overwrites the other."""
    frame, feats, _ = align(two_stations(tmp_path))
    for c in feats:
        assert f"AAA__{c}" in frame.columns and f"BBB__{c}" in frame.columns


def test_stations_extracted_with_different_configs_are_refused(tmp_path):
    """Different feature sets are not comparable, and concatenating them would
    line up columns that mean different things."""
    per = two_stations(tmp_path)
    per["BBB"] = per["BBB"].drop(columns=[c for c in per["BBB"].columns
                                          if c.startswith("EN_CROSS_CORR")])
    with pytest.raises(SystemExit, match="different feature set"):
        align(per)
