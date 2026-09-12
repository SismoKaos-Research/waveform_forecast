"""Windows of multi-station hourly features, standardized against the train split.

Not a runnable script -- imported only.

One sample is `seq_hours` consecutive hours ending at an index, shaped
(time, stations, features), with the presence mask alongside it and the label at
the window's last hour.

**Standardization is per (station, feature) and fitted on present rows only.**
Per-station because site response is a property of the station, not the
earthquake -- pooling z-scores from one shared mean would make a permanently
noisier station look permanently more active. Present rows only because an
absent hour is NaN, and letting NaN into the mean makes every statistic NaN.

**NaN is filled with 0 after standardizing, and that is safe only because of the
mask.** Zero here means "the standardized mean", which is exactly the reading
that would be indistinguishable from a real quiet hour -- which is why the model
never sees these cells: `MaskedStationPool` zeroes masked embeddings before
weighting them. The fill exists because `NaN * 0` is NaN, so a masked-out cell
left as NaN still poisons the pooled vector and every gradient after it.
"""
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class StackedStations:
    """A (hours, stations, ...) view over one array per station.

    The raw tensor is ~4.3 GB per station, so the station axis is stacked one
    window at a time instead of being materialised. Slicing on the hour axis is
    the only access `MultiStationWindows` needs, and that touches `seq_hours`
    rows per sample rather than the whole archive.
    """

    def __init__(self, arrays):
        self.arrays = list(arrays)
        a0 = self.arrays[0]
        self.shape = (a0.shape[0], len(self.arrays)) + tuple(a0.shape[1:])
        self.ndim = len(self.shape)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        """Indexes the HOUR axis and stacks the stations into axis 1."""
        return np.stack([np.asarray(a[key]) for a in self.arrays], axis=1)

    def select(self, idx):
        """The sub-view holding only these station positions."""
        return StackedStations([self.arrays[i] for i in np.asarray(idx)])


def load_waveform_tensor(path, stations):
    """Memmaps `waveform-forecast waveforms` output for the given stations.

    Returns:
        (x, present, hour_index) where `x` is a `StackedStations` of shape
        (hours, stations, 3, samples_per_hour).

    Raises:
        SystemExit: If the directory or a station's array is missing.
    """
    import sys
    from pathlib import Path
    d = Path(path)
    idx_path = d / "index.parquet"
    if not idx_path.exists():
        sys.exit(f"[ERROR] {idx_path} does not exist. Build it with "
                 f"`waveform-forecast waveforms --station CODE=DIR --out {d}`.")
    idx = pd.read_parquet(idx_path)
    arrays, mask = [], []
    for s in stations:
        f = d / f"{s}.npy"
        if not f.exists():
            have = sorted(p.stem for p in d.glob("*.npy"))
            sys.exit(f"[ERROR] {f} does not exist; {d} holds {have}")
        if f"present_{s}" not in idx.columns:
            sys.exit(f"[ERROR] {idx_path} has no present_{s} column")
        arrays.append(np.load(f, mmap_mode="r"))
        mask.append(idx[f"present_{s}"].to_numpy(dtype=bool))
    return (StackedStations(arrays), np.stack(mask, axis=1),
            pd.DatetimeIndex(idx.index))


def station_feature_columns(frame, stations):
    """The `<STATION>__<FEATURE>` columns, per station, in one consistent order.

    Returns:
        (columns, feature_names) where `columns` is {station: [col, ...]} and
        every station's list is the same features in the same order -- the
        station axis is only stackable if it is.

    Raises:
        ValueError: If a station is missing from the frame, or the stations do
            not share a feature set.
    """
    cols, feats = {}, None
    for s in stations:
        pre = f"{s}__"
        got = [c for c in frame.columns if c.startswith(pre)]
        if not got:
            raise ValueError(
                f"station {s!r} has no columns in this table; it holds "
                f"{sorted({c.split('__')[0] for c in frame.columns if '__' in c})}")
        names = [c[len(pre):] for c in got]
        if feats is None:
            feats = names
        elif names != feats:
            raise ValueError(
                f"station {s!r} carries a different feature set than "
                f"{stations[0]!r}; extract every station with the same config")
        cols[s] = [f"{pre}{f}" for f in feats]
    return cols, feats


def stack(frame, stations):
    """The frame as (hours, stations, features) plus its (hours, stations) mask."""
    cols, feats = station_feature_columns(frame, stations)
    x = np.stack([frame[cols[s]].to_numpy(dtype=np.float32) for s in stations], axis=1)
    present = np.stack([frame[f"present_{s}"].to_numpy(dtype=bool) for s in stations],
                       axis=1)
    return x, present, feats


def fit_stats(x, present, indices, seq_hours, all_station_rows=True):
    """Per-(station, feature) mean and std over the training windows' present hours.

    Sampled across the whole training split rather than its opening rows: a
    trailing-window feature's lookback is still filling at the start of an
    archive, and standardizing by statistics taken there understated the true
    spread by up to 52x in this project's single-station work -- producing
    z-scores over 100 and a model whose best checkpoint was the untrained one.

    **`all_station_rows` is what makes a spatial holdout standardizable.** With
    a held-out ZONE, the test stations appear in no training window at all, so
    the per-station statistics for them would be all-NaN and fall back to the
    identity -- their features would reach the model raw while the training
    stations' arrived z-scored, and the run would be measuring a scaling
    mismatch rather than transfer.

    The fix is not to pool one shared statistic across stations (site response
    is a property of the station, which is why these are per-station in the
    first place), nor to fit the held-out station on its own test-period data
    (that is the evaluation split's distribution entering the model). It is to
    fit every station on **its own hours inside the training TIME window**. The
    Marmara stations were recording throughout the Aegean training period; they
    were simply not the labelled zone. Those hours are theirs, they are in the
    past relative to the test block, and they carry no label at all -- so using
    them costs nothing and preserves the per-station scaling.

    Args:
        x: (hours, stations, features), NaN where absent.
        present: (hours, stations) bool.
        indices: TRAIN window end-indices.
        seq_hours: Hours per window.
        all_station_rows: Fit every station over the training window's hour
            RANGE, rather than only over the hours its own windows covered.
            Identical for a station whose windows span the split; the
            difference is exactly the held-out-zone case. False restores the
            pre-spatial behaviour.

    Returns:
        (mu, sd), each (stations, features) float32.
    """
    take = indices[np.linspace(0, len(indices) - 1,
                               min(500, len(indices))).astype(int)]
    rows = np.unique(np.concatenate(
        [np.arange(max(0, i - seq_hours + 1), i + 1) for i in take]))
    # The raw tensor is ~2.6 MB per (hour, 2 stations), so the contiguous train
    # range below would be several GB. Statistics over a few hundred sampled
    # hours are already far tighter than the between-hour variation they
    # describe, so cap the read rather than the accuracy.
    raw = getattr(x, "ndim", 0) == 4
    if all_station_rows and len(rows):
        # The contiguous hour span the training windows cover. A station absent
        # from the training ZONE still has its own readings in here, and those
        # are what standardize it. Bounded by the train window's own extent, so
        # nothing after the split boundary is ever touched.
        rows = np.arange(int(rows.min()), int(rows.max()) + 1)
    if raw and len(rows) > 400:
        rows = rows[np.linspace(0, len(rows) - 1, 400).astype(int)]
    sub, msk = x[rows], present[rows]
    if sub.ndim == 4:
        # Raw arm: (rows, stations, channels, samples). Statistics are per
        # (station, CHANNEL) -- pooled over samples, not per sample. A per-sample
        # mean would be 18,000 numbers per channel describing nothing but where
        # in the hour a sample sat, and would standardize away the amplitude
        # that is the entire signal. Kept with a trailing axis so it broadcasts
        # back over samples.
        m4 = msk[..., None, None]
        wide = np.where(m4, sub, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mu = np.nanmean(wide, axis=(0, 3), keepdims=False)
            sd = np.nanstd(wide, axis=(0, 3), keepdims=False)
        mu = np.where(np.isfinite(mu), mu, 0.0).astype(np.float32)[..., None]
        sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0).astype(np.float32)[..., None]
        return mu, sd
    wide = np.where(msk[..., None], sub, np.nan)
    # A station can be absent for an ENTIRE training split and that is not an
    # error -- GCAM's archive ends 2024-12, so any later fold has none of it.
    # nanmean of an all-NaN column is NaN with a RuntimeWarning; the identity
    # (0, 1) is the right fill, because the mask means those cells never reach
    # the model anyway. Silencing the warning here rather than globally keeps
    # it meaningful everywhere else.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(wide, axis=0)
        sd = np.nanstd(wide, axis=0)
    absent = ~np.isfinite(mu).all(axis=-1)
    if absent.any():
        print(f"    [stats] {int(absent.sum())} station(s) absent for the whole "
              f"training split; standardized with the identity, and masked out "
              f"at every hour anyway")
        if all_station_rows:
            # In a spatial holdout this is no longer harmless. A held-out
            # station that is also absent from the training TIME window has no
            # honest statistics available, and the identity means it reaches
            # the model unstandardized rather than masked away.
            print(f"          [!] with a held-out zone, a station absent here "
                  f"is NOT masked at test time --\n              it will be fed "
                  f"raw while the training stations were z-scored. Choose a "
                  f"train\n              window that overlaps the test "
                  f"stations' archive.")
    mu = np.where(np.isfinite(mu), mu, 0.0).astype(np.float32)
    sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0).astype(np.float32)
    return mu, sd


class MultiStationWindows(Dataset):
    """Windows of (time, stations, features), the mask, and the label."""

    def __init__(self, x, present, labels, seq_hours, indices, stats,
                 station_idx=None):
        """Builds the dataset.

        Args:
            x: (hours, stations, features) float array, NaN where absent.
            present: (hours, stations) bool array.
            labels: (hours,) int array.
            seq_hours: Hours per window.
            indices: Window end-indices.
            stats: (mu, sd) from `fit_stats` on the TRAINING split. Val and test
                must reuse them; fitting their own would let the evaluation
                split's distribution into the model.
            station_idx: Positions on the station axis this split uses, or None
                for all of them. A spatial holdout trains on one zone's slice
                and tests on another's; `MaskedStationPool` is attention over
                the station axis and shares one projection across stations, so
                the two slices need not be the same width and no part of the
                model is tied to a particular station's position.
        """
        self.x, self.present, self.labels = x, present, labels
        self.seq_hours, self.indices = seq_hours, indices
        mu, sd = stats
        self.station_idx = station_idx
        if station_idx is not None:
            k = np.asarray(station_idx, dtype=np.int64)
            # Slice x/present/stats once here rather than per __getitem__: the
            # same three lines in the hot path cost a fancy-index copy on every
            # sample, and the arrays are views until something writes to them.
            # `StackedStations` selects whole per-station arrays instead, since
            # its station axis does not exist until a window is requested.
            self.x = x.select(k) if isinstance(x, StackedStations) else x[:, k]
            self.present = present[:, k]
            mu, sd = mu[k], sd[k]
        self.mu, self.sd = mu, sd

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """Returns (window, present, label) for one sample."""
        end = self.indices[idx]
        start = end - self.seq_hours + 1
        w = (self.x[start:end + 1] - self.mu) / self.sd
        m = self.present[start:end + 1]
        # The mask is (time, stations); the window is (time, stations, features)
        # on the feature arm and (time, stations, channels, samples) on the raw
        # one. Broadcast by the difference rather than a fixed `[..., None]`, so
        # one expression covers both and the raw arm cannot silently mask the
        # wrong axis.
        me = m.reshape(m.shape + (1,) * (w.ndim - m.ndim))
        # See the module docstring: masked cells must be finite, not NaN, or
        # `NaN * 0` in the pool takes the whole batch with it. The mask is what
        # keeps this zero from being read as a real average reading.
        w = np.where(me, np.nan_to_num(w, nan=0.0), 0.0)
        return (torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(m)),
                torch.tensor(self.labels[end], dtype=torch.float32))
