"""Extractor output in, one aligned multi-station hourly table out.

    waveform-forecast features \
        --station MANT=../extracted_features/mant_features.parquet \
        --station DEMI=../extracted_features/demi_features.parquet \
        --out hourly_mant_demi.parquet

**This does not compute features.** `sismokaos-cli` (Rust) and
`Sismokaos-featureExtract` (Python) do that, and they agree on a column scheme:
`Pencere_ID`, `Zaman_Dk`, then `{E,N,Z}_STA_LTA_Max`, `{E,N,Z}_HJORTH_ACTIVITY`,
`EN_CROSS_CORR` and the rest, each also appearing as a `_DEV` column holding its
diff against the previous window. Reimplementing that here would give two
definitions of the same feature and no way to tell which produced a number.

What this does is the part neither extractor does: put several stations on one
clock.

**Three things have to happen for that, and each is a place to go wrong.**

*Aggregation.* The extractors emit a window every `step_sec` (50 s in the
shipped config, verified at a 50.0 s median on the BODT parquet), and the label
is hourly. So ~72 windows collapse into one row. The statistic matters: a
transient is what STA/LTA measures, and averaging it over an hour is how you
lose the only thing it was computing. `max` is kept alongside `mean` for exactly
that reason.

*Alignment.* Stations do not share a clock or a span. MANT covers 756 days, DEMI
563, ELBA 252, GCAM 189; all three Aegean stations overlap on 80 days and all
four on 29. Reindexing onto the union rather than the intersection is what makes
the other ~700 days usable.

*Presence.* An hour with no data for a station is not an hour of quiet. The
output carries `present_<STATION>` per row, and the model pools over it -- see
`model.MaskedStationPool`. Filling those hours with zeros or a column mean would
hand the model a reading it cannot distinguish from a real one.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NAME = "features"
HELP = "aggregate extractor output into one aligned multi-station hourly table"

ID_COLUMNS = ("Pencere_ID", "Zaman_Dk")

# `max` is not decoration. STA/LTA and PEAK measure a transient; their hourly
# mean is dominated by the ~72 quiet windows around it and reports roughly the
# noise floor. Whatever else is dropped, the max of those has to survive.
DEFAULT_AGGS = ("mean", "std", "max")


def read_extractor_output(path):
    """Reads one station's extractor output, parquet or CSV, by extension.

    Returns:
        DataFrame indexed by UTC timestamp, id columns dropped.

    Raises:
        SystemExit: If the file has no `Zaman_Dk`, which every extractor writes
            and without which the rows cannot be placed on a clock at all.
    """
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "Zaman_Dk" not in df.columns:
        sys.exit(f"[ERROR] {path} has no `Zaman_Dk` column. Both extractors "
                 f"write it; got {list(df.columns)[:8]}...")
    # Zaman_Dk is MINUTES since the Unix epoch, not seconds. Read as seconds it
    # lands in 1970 and every label joins against nothing -- silently, since an
    # empty join is an empty table rather than an error.
    df = df.assign(_t=pd.to_datetime(df["Zaman_Dk"] * 60.0, unit="s", utc=True))
    df = df.drop(columns=[c for c in ID_COLUMNS if c in df.columns])
    return df.set_index("_t").sort_index()


def to_hourly(df, aggs=DEFAULT_AGGS):
    """Collapses ~72 windows an hour into one row per hour.

    Args:
        df: Timestamp-indexed feature frame from `read_extractor_output`.
        aggs: Statistics to keep per feature.

    Returns:
        DataFrame indexed by hour start, columns `<FEATURE>_<agg>`, plus
        `n_windows` -- the count the hour was built from, which is what makes a
        thin hour visible instead of merely quiet.
    """
    num = df.select_dtypes(include=[np.number])
    g = num.groupby(num.index.floor("h"))
    out = g.agg(list(aggs))
    out.columns = [f"{c}_{a}" for c, a in out.columns]
    # Count the windows that CARRIED something, not the rows that existed. The
    # extractors write an all-NaN row for a window they could not compute --
    #0.06% of the BODT parquet, concentrated on gaps -- and an hour made
    # entirely of those is not a thin observation, it is no observation. Sizing
    # by `g.size()` let such an hour pass the --min-windows check and reach the
    # model flagged present with every feature NaN.
    out["n_windows"] = num.notna().any(axis=1).groupby(num.index.floor("h")).sum()
    out.index.name = "hour"
    return out


def align(per_station, min_windows=1):
    """Puts every station on one hourly index and records who was actually there.

    Args:
        per_station: {station: hourly DataFrame}.
        min_windows: An hour built from fewer windows than this counts as
            absent. A single 50 s window is not an hour of observation, and
            treating it as one lets the tail of a chunk look like a real
            reading.

    Returns:
        (frame, feature_names, stations). `frame` is indexed by hour and holds
        `<STATION>__<FEATURE>` columns plus one `present_<STATION>` per station.
        Absent cells are NaN, never 0 -- the mask is the only thing that says
        which is which, so a NaN that silently became a zero would be
        indistinguishable from a genuinely average reading.
    """
    stations = sorted(per_station)
    index = None
    for s in stations:
        index = per_station[s].index if index is None else index.union(per_station[s].index)
    index = pd.DatetimeIndex(index).sort_values()

    feats = None
    frames = []
    for s in stations:
        h = per_station[s]
        cols = [c for c in h.columns if c != "n_windows"]
        # Both conditions, not just the count: an hour can clear --min-windows
        # and still aggregate to all-NaN if every one of those windows failed a
        # different feature. Presence has to mean "there is something here".
        enough = (h["n_windows"] >= min_windows) & h[cols].notna().any(axis=1)
        if feats is None:
            feats = cols
        elif cols != feats:
            missing = set(feats) ^ set(cols)
            sys.exit(f"[ERROR] {s} has a different feature set than the first "
                     f"station; {len(missing)} column(s) differ, e.g. "
                     f"{sorted(missing)[:4]}. Extract every station with the "
                     f"same config, or the stations are not comparable.")
        block = h.loc[enough, cols].reindex(index)
        block.columns = [f"{s}__{c}" for c in cols]
        block[f"present_{s}"] = enough.reindex(index, fill_value=False)
        frames.append(block)
    return pd.concat(frames, axis=1), feats, stations


def add_args(p):
    p.add_argument("--station", action="append", required=True, metavar="CODE=PATH",
                   help="repeatable, e.g. MANT=feats/mant.parquet. The code is "
                        "what --stations selects by at train time.")
    p.add_argument("--agg", default=",".join(DEFAULT_AGGS),
                   help="comma-separated statistics per feature. `max` matters: "
                        "STA/LTA and PEAK measure transients, and their hourly "
                        "mean reports the noise floor around one.")
    p.add_argument("--min-windows", type=int, default=8,
                   help="hours built from fewer extractor windows than this are "
                        "marked absent rather than kept as a thin reading "
                        "(a full hour is ~72 windows at step_sec 50)")
    p.add_argument("--drop-dev", action="store_true",
                   help="drop the extractors' `_DEV` (inter-window diff) "
                        "columns, halving the width")
    p.add_argument("--out", required=True)
    return p


def run(args):
    """Reads each station's extractor output, aggregates, aligns, writes."""
    aggs = tuple(a.strip() for a in args.agg.split(",") if a.strip())
    per_station = {}
    for spec in args.station:
        if "=" not in spec:
            sys.exit(f"[ERROR] --station wants CODE=PATH, got {spec!r}")
        code, path = spec.split("=", 1)
        df = read_extractor_output(path)
        if args.drop_dev:
            df = df[[c for c in df.columns if not c.endswith("_DEV")]]
        hourly = to_hourly(df, aggs)
        per_station[code] = hourly
        print(f"  {code:6s} {len(df):>9,} windows  {df.index.min()} -> "
              f"{df.index.max()}  -> {len(hourly):,} hours")

    frame, feats, stations = align(per_station, args.min_windows)

    print(f"\n  aligned onto {len(frame):,} hourly rows, "
          f"{frame.index.min()} -> {frame.index.max()}")
    print(f"  {len(feats)} feature column(s) per station, {len(stations)} station(s)")
    print("\n  presence:")
    present = frame[[f"present_{s}" for s in stations]]
    for s in stations:
        n = int(present[f"present_{s}"].sum())
        print(f"    {s:6s} {n:>6,} hours ({100 * n / len(frame):5.1f}%)")
    both = int(present.all(axis=1).sum())
    none = int((~present.any(axis=1)).sum())
    print(f"    {'all':6s} {both:>6,} hours ({100 * both / len(frame):5.1f}%)"
          f"   <- what an intersection-only model would get")
    print(f"    {'any':6s} {len(frame) - none:>6,} hours "
          f"({100 * (len(frame) - none) / len(frame):5.1f}%)"
          f"   <- what the masked model trains on")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out)
    print(f"\n  wrote {out}  ({out.stat().st_size / 1e6:.0f} MB)")
    return 0


def main():
    p = argparse.ArgumentParser(prog="waveform-forecast features", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(p)
    return run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
