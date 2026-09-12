"""Which events belong to which stations, and what happens when they don't.

The four archives on disk are two tectonic settings, not one. MANT and DEMI sit
in the Aegean extensional province at 38-39N; ELBA and SEMS sit on the Marmara
segment of the North Anatolian Fault at 40.9-41.1N -- **north of
`catalog.AEGEAN_BBOX`, which stops at 40N.**

So under the region-wide label the Marmara pair was being scored on Aegean
earthquakes 221-295 km away. Nothing errored; the stations simply carried a
label their own seismograms have no view of. That is the failure this module
exists to make impossible, and the first test here is the one that would have
caught it.
"""
import numpy as np
import pandas as pd
import pytest

from waveform_forecast.catalog import AEGEAN_BBOX, haversine_km
from waveform_forecast.regions import (KNOWN_STATIONS, ZONES, describe_zone,
                                       in_window, load_events_near,
                                       load_station_coords, resolve_stations)


@pytest.fixture
def catalog(tmp_path):
    """A catalogue with one event beside each zone, and one far from both."""
    rows = [
        # near MANT (38.4908, 28.5579) -- Aegean
        ("01/06/2024 00:00:00", 38.50, 28.56, 5.0),
        # near ELBA (41.1469, 28.4307) -- Marmara
        ("02/06/2024 00:00:00", 41.15, 28.43, 5.0),
        # near SEMS (40.8710, 29.7391) -- Marmara
        ("03/06/2024 00:00:00", 40.87, 29.74, 5.0),
        # Cyprus-ish: far from every station here
        ("04/06/2024 00:00:00", 35.00, 33.00, 6.0),
        # below any threshold used in these tests
        ("05/06/2024 00:00:00", 38.50, 28.56, 1.0),
    ]
    p = tmp_path / "cat.csv"
    pd.DataFrame(rows, columns=["Date", "Latitude", "Longitude",
                                "Magnitude"]).to_csv(p, index=False)
    return str(p)


# --- the bug this module exists for ----------------------------------------

def test_the_marmara_stations_are_outside_the_aegean_box():
    """The assumption the region-wide label silently rested on, stated as a test.

    `AEGEAN_BBOX` is the label's footprint. A station outside it is forecasting
    a region it does not observe, and nothing in the pipeline notices -- the
    join succeeds, the labels are all valid, the run completes. Pinning it here
    means adding a fifth station outside the box fails loudly instead.
    """
    lat0, lat1, lon0, lon1 = AEGEAN_BBOX
    inside = {c for c, (la, lo) in KNOWN_STATIONS.items()
              if lat0 <= la <= lat1 and lon0 <= lo <= lon1}
    assert inside == set(ZONES["aegean"])
    assert not (inside & set(ZONES["marmara"]))


def test_the_zones_are_far_enough_apart_to_be_a_holdout():
    """221+ km between the pairs, 63-114 km within them."""
    def d(a, b):
        (la0, lo0), (la1, lo1) = KNOWN_STATIONS[a], KNOWN_STATIONS[b]
        return haversine_km(la0, lo0, np.array([la1]), np.array([lo1]))[0]

    within = max(d(*ZONES["aegean"]), d(*ZONES["marmara"]))
    between = min(d(a, b) for a in ZONES["aegean"] for b in ZONES["marmara"])
    assert within < 150
    assert between > 200


# --- region-local loading --------------------------------------------------

def test_a_zone_gets_its_own_events_and_not_the_other_zones(catalog):
    coords = load_station_coords()
    aeg = [xy for _, xy in resolve_stations(["aegean"], coords)]
    mar = [xy for _, xy in resolve_stations(["marmara"], coords)]
    assert len(load_events_near(catalog, 4.5, aeg, 50)) == 1
    assert len(load_events_near(catalog, 4.5, mar, 50)) == 2


def test_no_bounding_box_is_applied(catalog):
    """The Marmara events are north of AEGEAN_BBOX and must still be found.

    `load_aegean_events` would return none of them. That is the difference.
    """
    coords = load_station_coords()
    mar = [xy for _, xy in resolve_stations(["marmara"], coords)]
    got = load_events_near(catalog, 4.5, mar, 50)
    assert len(got) == 2
    lat1 = AEGEAN_BBOX[1]
    assert all(KNOWN_STATIONS[c][0] > lat1 for c in ZONES["marmara"])


def test_the_radius_is_to_the_nearest_station_not_a_centroid(catalog):
    """ELBA and SEMS are 114 km apart; a centroid would sit 57 km from each.

    A tight radius around each station still catches both events. A radius
    measured from the midpoint would need to be ~57 km larger to do the same,
    so the two rules are not interchangeable.
    """
    coords = load_station_coords()
    mar = [xy for _, xy in resolve_stations(["marmara"], coords)]
    assert len(load_events_near(catalog, 4.5, mar, 20)) == 2
    mid = [((mar[0][0] + mar[1][0]) / 2, (mar[0][1] + mar[1][1]) / 2)]
    assert len(load_events_near(catalog, 4.5, mid, 20)) == 0


def test_the_magnitude_threshold_still_applies(catalog):
    aeg = [xy for _, xy in resolve_stations(["aegean"], load_station_coords())]
    assert len(load_events_near(catalog, 4.5, aeg, 50)) == 1
    assert len(load_events_near(catalog, 0.5, aeg, 50)) == 2


def test_no_radius_means_no_spatial_restriction(catalog):
    """`--label-radius-km` unset is the region-wide label, not an empty one."""
    aeg = [xy for _, xy in resolve_stations(["aegean"], load_station_coords())]
    assert len(load_events_near(catalog, 4.5, aeg, None)) == 4
    assert len(load_events_near(catalog, 4.5, aeg, 0)) == 4


def test_events_come_back_sorted(catalog):
    """Every label function downstream uses searchsorted on this array."""
    aeg = [xy for _, xy in resolve_stations(["aegean"], load_station_coords())]
    got = load_events_near(catalog, 0.5, aeg, None)
    assert list(got) == sorted(got)


# --- resolving stations ----------------------------------------------------

def test_a_zone_name_expands_to_its_stations():
    coords = load_station_coords()
    assert [c for c, _ in resolve_stations(["marmara"], coords)] == ["ELBA", "SEMS"]


def test_codes_and_zone_names_mix():
    coords = load_station_coords()
    got = [c for c, _ in resolve_stations(["MANT", "marmara"], coords)]
    assert got == ["MANT", "ELBA", "SEMS"]


def test_a_station_named_twice_appears_once():
    """Order preserved, duplicates dropped -- the station axis must be stackable."""
    coords = load_station_coords()
    got = [c for c, _ in resolve_stations(["MANT", "aegean", "MANT"], coords)]
    assert got == ["MANT", "DEMI"]


def test_an_unknown_station_is_refused_rather_than_dropped():
    """A missing coordinate would make the distance mask match nothing.

    That is indistinguishable from a genuinely aseismic zone, so it has to be
    an error rather than an empty result.
    """
    with pytest.raises(SystemExit):
        resolve_stations(["NOPE"], load_station_coords())


def test_the_station_table_overrides_the_fallback(tmp_path):
    p = tmp_path / "st.csv"
    pd.DataFrame([("TU", "MANT", 1.0, 2.0)],
                 columns=["Network", "Code", "Longitude", "Latitude"]
                 ).to_csv(p, index=False, encoding="utf-8-sig")
    assert load_station_coords(str(p))["MANT"] == (2.0, 1.0)


def test_a_missing_station_table_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        load_station_coords(str(tmp_path / "nope.csv"))


# --- reporting the effective sample size -----------------------------------

def test_only_events_inside_the_archive_span_are_counted():
    """The catalogue starts in 2000; the archives start in 2024.

    Counting catalogue-wide made a zone with 8 usable events report 39, which
    is exactly the effective-sample-size confusion this project keeps hitting.
    """
    hours = pd.date_range("2024-01-01", periods=24, freq="h")
    times = np.array(["2011-05-01T00", "2024-01-01T05", "2030-01-01T00"],
                     dtype="datetime64[ns]")
    assert len(in_window(times, hours)) == 1


def test_an_empty_catalogue_survives_the_count():
    hours = pd.date_range("2024-01-01", periods=4, freq="h")
    assert len(in_window(np.array([], dtype="datetime64[ns]"), hours)) == 0


def test_a_thin_zone_is_flagged(capsys):
    """Under 20 events in span, the report says so before anything trains."""
    hours = pd.date_range("2024-01-01", periods=48, freq="h")
    times = np.array(["2024-01-01T05"], dtype="datetime64[ns]")
    describe_zone("test zone", ["ELBA"], [(41.1, 28.4)], times, 150, hours)
    assert "small effective sample" in capsys.readouterr().out
