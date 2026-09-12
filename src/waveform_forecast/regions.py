"""Where the stations are, and which events belong to them.

Not a runnable script -- imported only.

**Why this exists beside `catalog.py` rather than inside it.** `catalog.py` is a
byte-identical port from `../forecast`; that is the whole basis for claiming this
project and `catalog_mlp` answer the same question. Its `load_aegean_events`
filters to `AEGEAN_BBOX` (36-40N, 25-30E) and that box is not a parameter. So
region-local labelling is added here instead, and the port stays untouched.

**The problem it solves.** The four stations on disk are not one seismic setting:

    MANT  38.4908N 28.5579E  Manisa/Kula      Aegean extensional province
    DEMI  39.0428N 28.7162E  Manisa/Demirci   Aegean extensional province
    ELBA  41.1469N 28.4307E  Istanbul/Catalca Marmara / North Anatolian Fault
    SEMS  40.8710N 29.7391E  Kocaeli/Korfez   Marmara / North Anatolian Fault

MANT-DEMI are 63 km apart and ELBA-SEMS 114 km, but the pairs are 221-295 km
from each other. And **ELBA and SEMS are north of 40N, i.e. outside
`AEGEAN_BBOX` entirely.** Under the region-wide label they were being asked to
forecast events in a different tectonic province, ~250 km away, that their own
seismograms have essentially no view of.

That is not a bug in `catalog.py`. It is a design assumption -- "every station
sits inside the labelled region" -- that silently stopped holding when stations
outside the box were added. The region-wide label is right for comparing against
`catalog_mlp` on Aegean stations, and wrong the moment a station leaves the box.

**The distance is to the NEAREST station in the zone, not to a centroid.** A
centroid of MANT and DEMI sits between them and is a place no instrument is; a
zone of two stations 114 km apart is two overlapping disks, not one big one.
`catalog.station_distance_mask` already takes the nearest-station minimum, and
this module keeps that convention so the two agree.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from waveform_forecast.catalog import haversine_km

# Read off `data_downloader/catalogs/istasyon_katalog.csv` (AFAD's station
# table) for the four stations with archives on disk. Inlined as a fallback so
# a run does not depend on a sibling repository's path, and because a silently
# missing coordinate would produce a distance mask that quietly matches nothing.
KNOWN_STATIONS = {
    "MANT": (38.4908, 28.5579),
    "DEMI": (39.0428, 28.7162),
    "ELBA": (41.1469, 28.4307),
    "SEMS": (40.8710, 29.7391),
}

# The two tectonic settings the archives actually sample. Named zones exist so a
# run can say `--test-stations` by intent rather than by listing codes, and so
# the pairing is recorded in one place instead of in each command line.
ZONES = {
    "aegean": ("MANT", "DEMI"),
    "marmara": ("ELBA", "SEMS"),
}


def load_station_coords(path=None):
    """Station code -> (lat, lon), from AFAD's station table or the fallback.

    Args:
        path: `istasyon_katalog.csv` (columns Network, Code, Longitude,
            Latitude, ...). None uses `KNOWN_STATIONS` alone.

    Returns:
        Dict of code -> (lat, lon). The file's entries win over the fallback,
        so a corrected coordinate upstream takes effect without an edit here.
    """
    coords = dict(KNOWN_STATIONS)
    if path is None:
        return coords
    p = Path(path)
    if not p.exists():
        sys.exit(f"[ERROR] station table {p} does not exist. Omit the flag to "
                 f"use the built-in coordinates for {sorted(KNOWN_STATIONS)}.")
    # utf-8-sig: the AFAD export carries a BOM, which turns the first column
    # name into '﻿Network' and makes a lookup by "Network" miss.
    df = pd.read_csv(p, encoding="utf-8-sig")
    need = {"Code", "Latitude", "Longitude"}
    if not need.issubset(df.columns):
        sys.exit(f"[ERROR] {p} lacks {sorted(need - set(df.columns))}; got "
                 f"{list(df.columns)[:6]}")
    for _, r in df.iterrows():
        coords[str(r["Code"]).strip()] = (float(r["Latitude"]),
                                          float(r["Longitude"]))
    return coords


def resolve_stations(names, coords):
    """Expands zone names and station codes into (code, (lat, lon)) pairs.

    Args:
        names: Station codes, or a `ZONES` key like "marmara", mixed freely.
        coords: From `load_station_coords`.

    Returns:
        List of (code, (lat, lon)), de-duplicated, order preserved.

    Raises:
        SystemExit: On an unknown code. A station whose coordinate is missing
            would otherwise silently drop out of the distance mask, and a zone
            that matched no events looks exactly like a quiet zone.
    """
    out, seen = [], set()
    for n in names:
        for code in ZONES.get(n.lower(), (n,)):
            if code in seen:
                continue
            if code not in coords:
                sys.exit(f"[ERROR] no coordinate for station {code!r}. Known: "
                         f"{sorted(coords)[:8]}{'...' if len(coords) > 8 else ''}. "
                         f"Pass --station-table to load AFAD's full table.")
            seen.add(code)
            out.append((code, coords[code]))
    return out


def load_events_near(catalog_path, min_magnitude, centers, max_dist_km):
    """Catalogue events within `max_dist_km` of the nearest of `centers`.

    The region-local counterpart to `catalog.load_aegean_events`. **No bounding
    box**: the box is what excludes the Marmara stations' own seismicity, and a
    zone defined by distance to its own instruments does not need one.

    Args:
        catalog_path: Catalogue CSV with Date/Latitude/Longitude/Magnitude.
        min_magnitude: Completeness/qualifying threshold.
        centers: [(lat, lon), ...] -- the zone's stations.
        max_dist_km: Radius around the nearest one. None or <= 0 means no
            spatial restriction at all, which is the region-wide label.

    Returns:
        Sorted array of event times (numpy datetime64), as
        `load_aegean_events` returns, so every downstream label function takes
        it unchanged.
    """
    cat = pd.read_csv(catalog_path)
    cat["dt"] = pd.to_datetime(cat["Date"], format="%d/%m/%Y %H:%M:%S",
                               errors="coerce")
    sel = cat[(cat.Magnitude >= min_magnitude) & cat.dt.notna()]
    if max_dist_km and max_dist_km > 0:
        if not centers:
            sys.exit("[ERROR] --label-radius-km needs at least one station to "
                     "measure from")
        lats = sel.Latitude.to_numpy(dtype=np.float64)
        lons = sel.Longitude.to_numpy(dtype=np.float64)
        best = None
        for lat, lon in centers:
            d = haversine_km(lat, lon, lats, lons)
            best = d if best is None else np.minimum(best, d)
        sel = sel[best <= max_dist_km]
    return np.sort(sel.dt.to_numpy())


def in_window(times, hour_index):
    """The subset of `times` inside the archive's own span.

    The catalogue runs from 2000; the archives run from 2024. A zone's
    catalogue-wide event count is therefore several times its usable one, and
    the usable one is what sets the effective sample size. Reporting the wrong
    one makes a zone with 8 usable events look like a zone with 39.
    """
    if not len(times) or not len(hour_index):
        return times[:0]
    lo = np.datetime64(hour_index[0])
    hi = np.datetime64(hour_index[-1])
    return times[(times >= lo) & (times <= hi)]


def describe_zone(label, codes, centers, times, radius_km, hour_index=None):
    """One console block saying what a zone is and how many events define it.

    The effective sample size of a rare-event label is the number of distinct
    EVENTS, not the number of hours, and this project has repeatedly been
    misled by the difference. So the count is printed where the zone is chosen,
    before anything is trained -- and it is the count inside the ARCHIVE's span,
    not the catalogue's, because events from 2011 label no hour here.
    """
    print(f"  {label}: {'+'.join(codes)}")
    for code, (lat, lon) in zip(codes, centers):
        print(f"      {code:6s} {lat:8.4f}N {lon:8.4f}E")
    span = (f"within {radius_km:g} km of the nearest" if radius_km
            else "region-wide (no distance restriction)")
    if hour_index is None:
        print(f"      {len(times):,} qualifying event(s) in the catalogue, {span}")
        return
    here = in_window(times, hour_index)
    print(f"      {len(here):,} qualifying event(s) inside the archive span, "
          f"{span}\n          ({len(times):,} in the catalogue overall; the "
          f"earlier ones still set `days since previous`)")
    if len(here) < 20:
        print(f"      [!] {len(here)} events is a small effective sample. "
              f"Aftershock sequences\n          make the independent count "
              f"smaller still -- read the per-fold spread, not the mean.")
