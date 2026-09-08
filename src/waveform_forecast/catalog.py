"""Loading the earthquake catalogue, and turning it into hourly labels.

Not a runnable script -- imported only.

This is the whole input to the project. There is no waveform archive here: the
2026-08-30 experiment put the waveform and chaotic-feature arms below the
persistence floor, and concluded that "the forecasting signal in this project
comes from the earthquake catalogue, not from the seismogram." What survived
that boundary is what this module reads.

Ported from `cnn_earthquake/src/sismokaos/catalog.py` -- the seven functions
catalog_mlp uses, of its twenty. Bodies are unchanged; they produced the
published figures.
"""
import numpy as np
import pandas as pd

# lat0, lat1, lon0, lon1
AEGEAN_BBOX = (36.0, 40.0, 25.0, 30.0)

STATION_COORDS = {"BODT": (37.0622, 27.3103), "DAT": (36.7308, 27.5767)}


def haversine_km(lat0: float, lon0: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Great-circle distance in km from one point to an array of points."""
    r = 6371.0
    la1, lo1 = np.radians(lat0), np.radians(lon0)
    la2, lo2 = np.radians(lats), np.radians(lons)
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

def station_distance_mask(lats: np.ndarray, lons: np.ndarray, stations, max_dist_km: float):
    """Boolean mask for events within `max_dist_km` of the NEAREST named station."""
    if not max_dist_km or max_dist_km <= 0:
        return np.ones(len(lats), dtype=bool)
    best = None
    for s in stations:
        lat, lon = STATION_COORDS[s] if isinstance(s, str) else s
        d = haversine_km(lat, lon, lats, lons)
        best = d if best is None else np.minimum(best, d)
    return best <= max_dist_km

def load_aegean_events(catalog_path: str, min_magnitude: float = 4.5,
                       stations=None, max_dist_km: float = None) -> np.ndarray:
    """Loads catalog events within the Aegean bounding box at or above a magnitude."""
    cat = pd.read_csv(catalog_path)
    cat["dt"] = pd.to_datetime(cat["Date"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    lat0, lat1, lon0, lon1 = AEGEAN_BBOX
    aegean = cat[(cat.Latitude.between(lat0, lat1)) & (cat.Longitude.between(lon0, lon1)) &
                (cat.Magnitude >= min_magnitude) & cat.dt.notna()]
    if stations and max_dist_km:
        aegean = aegean[station_distance_mask(aegean.Latitude.to_numpy(),
                                            aegean.Longitude.to_numpy(),
                                            stations, max_dist_km)]
    return np.sort(aegean.dt.to_numpy())

def load_aegean_events_with_location(catalog_path: str, min_magnitude: float = 3.0,
                                     stations=None, max_dist_km: float = None):
    """Loads catalog events (times, magnitudes, AND lat/lon) within the Aegean bbox.

    Companion to `load_aegean_events_with_magnitude`, adding coordinates for
    features that need event location -- nearest-neighbour distance
    (Zaliapin & Ben-Zion) and spatial Shannon entropy, both from Convertito
    et al. 2024 (Sci. Rep. 14:2964).

    Args:
        catalog_path: Path to a catalog CSV with 'Date', 'Latitude',
            'Longitude', 'Magnitude' columns (data_large.csv format).
        min_magnitude: Minimum magnitude to include (completeness
            threshold for the returned "background" catalog).

    Returns:
        Tuple of (times, magnitudes, lats, lons), all sorted by time, same order.
    """
    cat = pd.read_csv(catalog_path)
    cat["dt"] = pd.to_datetime(cat["Date"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    lat0, lat1, lon0, lon1 = AEGEAN_BBOX
    aegean = cat[(cat.Latitude.between(lat0, lat1)) & (cat.Longitude.between(lon0, lon1)) &
                (cat.Magnitude >= min_magnitude) & cat.dt.notna()].sort_values("dt")
    if stations and max_dist_km:
        aegean = aegean[station_distance_mask(aegean.Latitude.to_numpy(),
                                              aegean.Longitude.to_numpy(),
                                              stations, max_dist_km)]
    return (aegean.dt.to_numpy(), aegean.Magnitude.to_numpy(dtype=np.float64),
           aegean.Latitude.to_numpy(dtype=np.float64), aegean.Longitude.to_numpy(dtype=np.float64))

def count_events_in_window(hourly_index: pd.DatetimeIndex, times: np.ndarray,
                           window_days: float, forward: bool) -> np.ndarray:
    """Counts events in a trailing or leading window around each hour."""
    t = hourly_index.to_numpy()
    w = np.timedelta64(int(round(window_days * 24)), "h")
    if forward:
        return (np.searchsorted(times, t + w, side="right")
                - np.searchsorted(times, t, side="right")).astype(np.int64)
    return (np.searchsorted(times, t, side="right")
            - np.searchsorted(times, t - w, side="right")).astype(np.int64)

def days_since_prev_major(hourly_index: pd.DatetimeIndex, major_times: np.ndarray) -> np.ndarray:
    """Computes days elapsed since the previous qualifying event, per hour."""
    t = hourly_index.to_numpy()
    out = np.full(len(t), np.nan)
    for i, ti in enumerate(t):
        prev = major_times[major_times < ti]
        if len(prev):
            out[i] = (ti - prev[-1]) / np.timedelta64(1, "D")
    return out

def label_hours_rate_change(hourly_index: pd.DatetimeIndex, rate_times: np.ndarray,
                            horizon_days: float, baseline_days: float = None):
    """Labels each hour with whether seismicity RATE will increase ("variant B").

    A different forecasting target from `label_hours`: instead of "does one
    rare M>=threshold event occur in the next horizon" (whose positive class,
    at M>=4.5, is driven by a handful of events per fold -- 4 in fold 1 --
    making the effective sample size far smaller than the hour count
    suggests), this asks "will the next window contain MORE events than the
    trailing window did".

    That is a rate/acceleration forecast, which is what ETAS-family models and
    CSEP evaluation actually target, and it uses a much lower magnitude
    threshold (typically M>=3.0), so the label is driven by ~10^3 events
    instead of ~10^1. It is also the quantity Convertito et al. 2024's
    beta-statistic measures -- but as the target itself rather than as a mask
    on a rare-event label (`label_hours_beta_precursor`), which is what made
    that earlier attempt fail.

    Note the natural baseline here is strongly ANTI-correlated: during an
    aftershock sequence a high trailing rate predicts a DECREASE (Omori
    decay). Score any model against `rate_persistence_auc`, not against 0.5.

    Args:
        hourly_index: Hour-start timestamps, one per sample.
        rate_times: Sorted array of event times defining the rate (e.g.
            M>=3.0 events -- a much lower threshold than the label-defining
            `major_times` used by `label_hours`).
        horizon_days: Length of the forward window being forecast.
        baseline_days: Length of the trailing comparison window. Defaults to
            `horizon_days` (a like-for-like comparison, so the label is a
            clean "up or down" with no window-length bias).

    Returns:
        Tuple of (labels, forward_counts, trailing_counts) -- labels is an
        int64 0/1 array (1 = rate increases), the counts are returned so
        callers can build the persistence floor and report diagnostics
        without recomputing them.
    """
    if baseline_days is None:
        baseline_days = horizon_days
    fwd = count_events_in_window(hourly_index, rate_times, horizon_days, forward=True)
    bwd = count_events_in_window(hourly_index, rate_times, baseline_days, forward=False)
    return (fwd > bwd).astype(np.int64), fwd, bwd

def label_hours(hourly_index: pd.DatetimeIndex, major_times: np.ndarray,
                horizon_days: float, feature_hours: float = 1.0) -> np.ndarray:
    """Labels each hour with whether a qualifying event occurs within the horizon.

    **The horizon starts when the features END, not when the hour starts.**
    `hourly_index` holds hour STARTS and the features for hour H are aggregated
    over [H, H+1h], so a horizon opening at H counts an event occurring inside
    the very window the model is shown. That event is visible in the features
    and labelled as future -- the model can read off the answer. It is one hour
    of a 720-hour horizon, so it inflates rather than invents, but it is the
    same window-end mistake `parse_hour_start` and the Zaman_Dk handling above
    were written to avoid.

    `feature_hours=0` restores the old behaviour, for reproducing a figure
    published before this. Every forecasting number in the repo predates it.

    Args:
        hourly_index: Hour-start timestamps, one per sample.
        major_times: Sorted qualifying event times.
        horizon_days: How far ahead to look. Fractional days are honoured.
        feature_hours: Length of the feature window opening at each index, i.e.
            how far past the index the model can already see.

    Returns:
        Int array, 1 where a qualifying event falls in the horizon.
    """
    # timedelta64 with an int day count silently truncated: --horizon-days 0.5
    # became a ZERO-day horizon and every label came out negative. Seconds keep
    # sub-day horizons meaningful.
    horizon = np.timedelta64(int(round(horizon_days * 86400)), "s")
    offset = np.timedelta64(int(round(feature_hours * 3600)), "s")
    t = hourly_index.to_numpy()
    labels = np.zeros(len(t), dtype=np.int64)
    for i, ti in enumerate(t):
        start = ti + offset
        fut = major_times[(major_times > start) & (major_times <= start + horizon)]
        labels[i] = int(len(fut) > 0)
    return labels

def truncate_to_reliable_catalog_end(hour_index: pd.DatetimeIndex, raw: np.ndarray,
                                     major_times: np.ndarray, buffer_days: float = 0):
    """Drops hours past the point where the catalog can no longer reliably inform labels."""
    cutoff = major_times[-1] - np.timedelta64(int(buffer_days * 24), "h")
    n_keep = int((hour_index.to_numpy() <= cutoff).sum())
    if n_keep < len(hour_index):
        print(f"  [!] catalog's last event is {major_times[-1]} -- truncating archive from "
               f"{hour_index[-1]} to {hour_index[n_keep - 1]} ({len(hour_index) - n_keep} hours "
               f"dropped, buffer={buffer_days:.0f}d) to avoid right-censoring the forward-looking "
               f"label near the archive's end.")
    return hour_index[:n_keep], raw[:n_keep]
