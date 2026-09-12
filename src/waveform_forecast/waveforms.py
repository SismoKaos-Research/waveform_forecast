"""miniSEED archives in, one decimated hourly waveform tensor out.

    waveform-forecast waveforms \
        --station MANT=../mseed/MANT --station DEMI=../mseed/DEMI \
        --rate 5 --out wav5

**This is the stage `--arm raw` was missing.** `blocks.RawWaveformEncoder` and
the `--arm raw` branch of the trainer have existed since the first commit, but
nothing ever produced the tensor they consume: `build_inputs` reads the hourly
FEATURE parquet, so `--arm raw` fed an hourly feature vector to a Conv1d
expecting three channels of samples and died on the first forward pass:

    RuntimeError: Given groups=1, weight of size [16, 3, 7], expected
    input[1, 3072, 1] to have 3 channels, but got 3072 channels instead

So every published "raw waveform" result in this project came from elsewhere;
nothing in this repository could produce one.

**Why a separate command rather than reading miniSEED at train time.** The
archive is 86 GB of 100 Hz HH? channels across 121 files, and one file holds
~2,000 traces because the recording is gappy. Decoding that once per epoch, per
seed, per fold is not a thing that finishes. Decimated to 5 Hz and float32 it is
4.3 GB per station -- small enough to memmap and index directly.

**What "absent" means here, and why it is not zero.** A gap in the record is not
a quiet hour, exactly as in `features`. An hour is present only if enough of it
was actually recorded (`--min-coverage`); everything else is written as zeros
AND marked absent, and `MaskedStationPool` then gives it no weight at all. The
zeros are never read as data; the mask is what says so.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NAME = "waveforms"
HELP = "decimate miniSEED archives into one aligned hourly waveform tensor"

CHANNELS = ("E", "N", "Z")


def channel_of(trace_id):
    """The component letter of a SEED channel code, or None.

    AFAD's archives are HHE/HHN/HHZ; other bands (BH?, EH?) use the same final
    letter, so the last character is the component and the first two are the
    band and instrument.
    """
    code = trace_id.rsplit(".", 1)[-1]
    return code[-1].upper() if code else None


def hour_floor(t):
    """numpy datetime64 hour-start for an obspy UTCDateTime."""
    return np.datetime64(int(t.timestamp) // 3600 * 3600, "s").astype("datetime64[ns]")


def decimate_to(data, src_rate, dst_rate):
    """Anti-alias filters and downsamples one trace's samples.

    Args:
        data: 1D float array.
        src_rate: Sampling rate the samples were recorded at.
        dst_rate: Target rate.

    Returns:
        The decimated array.

    A low-pass BEFORE downsampling, not a plain stride. Taking every 20th
    sample of a 100 Hz record folds everything above 2.5 Hz back down into the
    band that is kept, and the energy an STA/LTA responds to lives exactly
    there -- the aliased result looks like signal and is not.
    """
    from scipy.signal import decimate as sp_decimate
    factor = int(round(src_rate / dst_rate))
    if factor <= 1:
        return data.astype(np.float32)
    out = data.astype(np.float64)
    # scipy's IIR decimate is unstable above ~13x in one pass, and 100 -> 5 Hz
    # is 20x. Factorising keeps every stage inside its stable range.
    for f in _factorise(factor):
        out = sp_decimate(out, f, ftype="fir", zero_phase=True)
    return out.astype(np.float32)


def _factorise(n, limit=10):
    """`n` as factors no larger than `limit`, largest first."""
    out = []
    for f in (10, 8, 5, 4, 3, 2):
        while n % f == 0 and f <= limit:
            out.append(f)
            n //= f
    if n > 1:
        out.append(n)
    return out or [1]


def station_hours(paths, rate, hours_per_file=None, verbose=True):
    """Reads one station's archive into {hour: (3, samples)} plus a coverage count.

    Returns:
        (chunks, covered) where `chunks` maps an hour-start to a
        (3, rate*3600) float32 array and `covered` maps it to the number of
        samples actually written into that hour.
    """
    import obspy
    n = int(round(rate * 3600))
    chunks, covered = {}, {}
    for p in sorted(paths):
        if verbose:
            print(f"      {Path(p).name}", flush=True)
        st = obspy.read(str(p))
        for tr in st:
            comp = channel_of(tr.id)
            if comp not in CHANNELS:
                continue
            ci = CHANNELS.index(comp)
            sr = float(tr.stats.sampling_rate)
            if sr < rate:
                continue
            data = decimate_to(tr.data.astype(np.float64), sr, rate)
            # Place each decimated sample by ABSOLUTE time, not by counting from
            # the trace start: the traces are gapped, and a trace that begins
            # 37 s into an hour must land 37 s in, or every sample after the
            # first gap is written to the wrong place.
            t0 = tr.stats.starttime.timestamp
            idx = np.arange(len(data))
            secs = t0 + idx / rate
            h = (secs // 3600).astype(np.int64)
            off = ((secs - h * 3600) * rate).astype(np.int64)
            ok = (off >= 0) & (off < n)
            for hh in np.unique(h[ok]):
                m = ok & (h == hh)
                key = np.datetime64(int(hh) * 3600, "s").astype("datetime64[ns]")
                buf = chunks.get(key)
                if buf is None:
                    buf = np.zeros((3, n), dtype=np.float32)
                    chunks[key] = buf
                    covered[key] = 0
                buf[ci, off[m]] = data[m]
                covered[key] += int(m.sum())
        del st
    return chunks, covered


def add_args(p):
    p.add_argument("--station", action="append", required=True, metavar="CODE=DIR",
                   help="repeatable, e.g. MANT=../mseed/MANT. The directory's "
                        "*.mseed files are read in name order.")
    p.add_argument("--rate", type=float, default=5.0,
                   help="target sampling rate in Hz. 5 is the shape the "
                        "published single-station runs used, and what "
                        "RawWaveformEncoder's strides assume.")
    p.add_argument("--min-coverage", type=float, default=0.5,
                   help="an hour with less than this fraction of its samples "
                        "recorded is marked ABSENT rather than zero-padded into "
                        "a reading. A gap is not a quiet sensor.")
    p.add_argument("--start", default=None, help="YYYY-MM-DD, inclusive")
    p.add_argument("--end", default=None, help="YYYY-MM-DD, exclusive")
    p.add_argument("--out", required=True,
                   help="output directory: <CODE>.npy per station plus index.parquet")
    return p


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = int(round(args.rate * 3600))

    per_station = {}
    for spec in args.station:
        if "=" not in spec:
            sys.exit(f"[ERROR] --station wants CODE=DIR, got {spec!r}")
        code, d = spec.split("=", 1)
        paths = sorted(Path(d).glob("*.mseed"))
        if not paths:
            sys.exit(f"[ERROR] no *.mseed under {d}")
        print(f"  {code}: {len(paths)} file(s)", flush=True)
        chunks, covered = station_hours(paths, args.rate)
        per_station[code] = (chunks, covered)
        print(f"  {code}: {len(chunks):,} hour(s) touched", flush=True)

    index = None
    for code, (chunks, _) in per_station.items():
        k = pd.DatetimeIndex(sorted(chunks))
        index = k if index is None else index.union(k)
    if args.start:
        index = index[index >= pd.Timestamp(args.start)]
    if args.end:
        index = index[index < pd.Timestamp(args.end)]
    # A contiguous hourly grid, not just the hours that happen to carry data:
    # `MultiStationWindows` takes `seq_hours` CONSECUTIVE rows, so a union index
    # with holes in it would silently splice across a gap and call it a window.
    index = pd.date_range(index.min().floor("h"), index.max().floor("h"), freq="h")
    print(f"\n  grid: {len(index):,} hours, {index[0]} -> {index[-1]}")

    mask = {}
    for code, (chunks, covered) in per_station.items():
        path = out / f"{code}.npy"
        # float32, not float16. Raw seismometer counts run to ~10^5-10^6 and
        # float16 saturates at 65504: a first pass at this silently turned
        # 52,637 cells of a single 21-day file into +-inf, pinned at exactly
        # 65504. An inf then meets `NaN * 0`'s cousin in MaskedStationPool --
        # inf * 0 is NaN -- and takes the pooled vector, the loss and every
        # gradient with it. Halving the file size is not worth a silent
        # amplitude clip on the loudest hours, which are precisely the ones a
        # forecaster cares about.
        arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                        shape=(len(index), 3, n))
        present = np.zeros(len(index), dtype=bool)
        pos = {t: i for i, t in enumerate(index.to_numpy())}
        for t, buf in chunks.items():
            i = pos.get(t)
            if i is None:
                continue
            if covered[t] < args.min_coverage * n:
                continue
            arr[i] = buf
            present[i] = True
        arr.flush()
        del arr
        mask[f"present_{code}"] = present
        print(f"  {code:6s} {present.sum():>6,} hour(s) present "
              f"({100 * present.mean():5.1f}%)   -> {path.name} "
              f"({path.stat().st_size / 1e9:.2f} GB)")

    idx = pd.DataFrame(mask, index=index)
    idx.index.name = "hour"
    idx.to_parquet(out / "index.parquet")
    any_p = idx.to_numpy().any(axis=1).mean()
    all_p = idx.to_numpy().all(axis=1).mean()
    print(f"    {'any':6s} present {100 * any_p:5.1f}%   "
          f"<- what the masked model trains on")
    print(f"    {'all':6s} present {100 * all_p:5.1f}%   "
          f"<- what an intersection-only model would get")
    print(f"\n  wrote {out}/index.parquet")
    return 0


def main():
    p = argparse.ArgumentParser(prog="waveform-forecast waveforms",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(p)
    return run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
