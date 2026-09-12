# The data pipeline

From several stations' extractor output to the tensor the model sees.

```
feats/<STA>/<STA>_features.parquet     ~50 s windows, one row each, per station
        │
        │  waveform-forecast features
        │    read_extractor_output → to_hourly → align
        ▼
hourly.parquet    hour-indexed, <STA>__<FEATURE>_<agg> columns + present_<STA>
        │
        │  train.build_inputs → data.stack
        ▼
x        (hours, stations, features)  float32, NaN where absent
present  (hours, stations)            bool
labels   (hours,)                     from catalog.py
        │
        │  data.fit_stats (TRAIN split only) → MultiStationWindows
        ▼
one sample: (seq_hours, stations, features), (seq_hours, stations), label
```

## This project does not compute features

`sismokaos-cli` (Rust) and `Sismokaos-featureExtract` (Python) do, and they agree
on a column scheme: `Pencere_ID`, `Zaman_Dk`, then `{E,N,Z}_STA_LTA_Max`,
`{E,N,Z}_HJORTH_ACTIVITY`, `EN_CROSS_CORR` and the rest, each also appearing as a
`_DEV` column holding its diff against the previous window.

Defining them a second time here would give two versions of the same quantity and
no way to say which produced a number. `features` does the part neither extractor
does: put several stations on one clock.

## `waveform-forecast features`

Three things have to happen, and each is a place to go wrong.

### 1. Aggregation — ~72 windows into one row

The extractors emit a window every `step_sec` (50 s in the shipped config,
verified at a 50.0 s median on the BODT parquet); the label is hourly. So about
72 windows collapse into one row.

**The statistic matters.** A transient is what STA/LTA and PEAK measure, and
averaging over an hour is how you lose the only thing they were computing — the
mean is dominated by the ~72 quiet windows around the transient and reports
roughly the noise floor. `max` is kept alongside `mean` and `std` for exactly
that reason (`DEFAULT_AGGS = ("mean", "std", "max")`).

**`n_windows` counts windows that carried something, not rows that existed.** The
extractors write an all-NaN row for a window they could not compute — 0.06% of
the BODT parquet, concentrated on gaps. Sizing an hour by `groupby.size()` let an
hour made entirely of those pass `--min-windows` and reach the model flagged
present with every feature NaN.

**`Zaman_Dk` is MINUTES since the Unix epoch, not seconds.** Read as seconds it
lands in 1970 and every label joins against nothing — silently, because an empty
join is an empty table rather than an error.

### 2. Alignment — the union, not the intersection

Stations share neither a clock nor a span. Reindexing onto the **union** of their
hourly indices is what makes the ~700 non-overlapping days usable at all; see
[architecture.md](architecture.md) for the day counts.

If two stations carry different feature sets, `align` exits rather than
proceeding: they were extracted with different configs and are not comparable.

### 3. Presence — an absent hour is not a quiet hour

The output carries one `present_<STATION>` column per station, and the model
pools over it. An hour counts as present only when **both** conditions hold:

- it was built from at least `--min-windows` windows (default 8; a full hour is
  ~72). A single 50 s window is not an hour of observation, and treating it as one
  lets the tail of a chunk look like a real reading.
- at least one feature in it is non-NaN. An hour can clear the window count and
  still aggregate to all-NaN if every window failed a different feature.

**Absent cells are NaN, never 0.** The mask is the only thing that distinguishes
absence from a genuinely average reading, so a NaN that silently became a zero
would be indistinguishable from data.

### What it prints

```
  MANT    1,276,344 windows  2024-05-01 ... 2026-08-18  -> 17,721 hours
  DEMI      927,180 windows  2024-09-25 ... 2026-08-18  -> 12,880 hours

  aligned onto 20,136 hourly rows, 2024-04-23 ... 2026-08-09
  207 feature column(s) per station, 2 station(s)

  presence:
    MANT   17,721 hours ( 88.0%)
    DEMI   12,880 hours ( 64.0%)
    all    10,791 hours ( 53.6%)   <- what an intersection-only model would get
    any    19,814 hours ( 98.4%)   <- what the masked model trains on
```

That last pair is the argument for the whole design, restated on every run.

## Windowing and standardization (`data.py`)

One sample is `--seq-hours` consecutive hours **ending** at an index, shaped
`(time, stations, features)`, with its presence mask alongside and **the label at
the window's last hour**.

### Standardization is per (station, feature), fitted on present rows only

- **Per station**, because site response is a property of the station. Pooling
  z-scores from one shared mean would make a permanently noisier station look
  permanently more active.
- **Present rows only**, because an absent hour is NaN and letting NaN into the
  mean makes every statistic NaN.
- **Fitted on the TRAIN split**; val and test reuse those statistics. Fitting
  their own would let the evaluation split's distribution into the model.

`fit_stats` samples up to 500 window end-indices spread **across the whole
training split**, not its opening rows. A trailing-window feature's lookback is
still filling at the start of an archive, and standardizing by statistics taken
there understated the true spread by up to 52× in this project's single-station
work — producing z-scores over 100 and a model whose best checkpoint was the
untrained one.

A station can be absent for an entire training split (GCAM's archive ends
2024-12, so any later fold has none of it). `nanmean` of an all-NaN column is NaN;
the identity `(mu=0, sd=1)` is the right fill, because the mask means those cells
never reach the model anyway. The run says so:

```
    [stats] 1 station(s) absent for the whole training split; standardized with
            the identity, and masked out at every hour anyway
```

### NaN is filled with 0 *after* standardizing

Zero here means "the standardized mean" — exactly the reading that would be
indistinguishable from a real quiet hour. It is safe only because the model never
sees those cells: `MaskedStationPool` zeroes masked embeddings before weighting
them. The fill exists because `NaN × 0` is NaN, so a masked-out cell left as NaN
still poisons the pooled vector and every gradient after it.

## Feature selection

The aligned table is ~200 columns **per station** (207 on the four-station
archive), while the positive class is a handful of distinct events per fold. So
`--keep-features` is usually not optional:

```
--keep-features Z_STA_LTA_Max_max EN_CROSS_CORR_mean
```

Names are the aggregated ones (`<FEATURE>_<agg>`), without the station prefix,
and an unknown name is an error rather than a silent no-op.

`--drop-dev` at `features` time drops the extractors' `_DEV` inter-window-diff
columns, halving the width before anything else sees them.

<a id="the-raw-tensor"></a>

## The raw tensor (`waveform-forecast waveforms`)

What `--arm raw` consumes, and the stage that did not exist until it was built.
86 GB of gapped 100 Hz miniSEED across 121 files becomes one hourly tensor of
shape (hours, 3, `rate`x3600), 4.3 GB per station at 5 Hz, memmapped and stacked
one window at a time by `data.StackedStations`.

Decoding miniSEED at train time is not an option — one file holds ~2,000 traces
because the recording is gappy, and it would be re-decoded every epoch, seed and
fold.

Three things in it are load-bearing:

**The decimation low-passes before it downsamples.** A plain `data[::20]` folds
everything above the new 2.5 Hz Nyquist back into the band that is kept, and
2–3 Hz is exactly where an STA/LTA responds — aliased noise that looks like
signal. Verified in `tests/test_waveforms.py`: 1 Hz survives at 0.95, 32 Hz drops
to 0.0000, while a stride leaves it at 1.0 folded to 2 Hz. 100 → 5 Hz is 20x,
past where `scipy.signal.decimate` is stable in one pass, so it runs as 10x
then 2x.

**Samples are placed by absolute time**, not by counting from each trace's
start. A trace beginning 37 s into an hour must land 37 s in; counting
misplaces everything after the first gap.

**The tensor is float32, not float16.** Raw counts reach 3.5e6 on MANT and
float16 saturates at 65,504. A first pass silently pinned 52,637 cells of a
single 21-day file to ±inf — no error, just clipped amplitudes on the loudest
hours, which are the ones a forecaster cares about. And `inf × 0` is NaN, so it
would have poisoned the pool exactly like the NaN bug the mask exists to
prevent, arriving through the amplitude instead of the gap.

Presence works as in `features`: an hour with less than `--min-coverage` of its
samples recorded is written as zeros **and marked absent**. The zeros are never
read as data; the mask is what says so.

Statistics for the raw arm are per **(station, channel)**, pooled over samples —
a per-sample mean would be 18,000 numbers describing where in the hour a sample
sat, and would standardize away the amplitude that is the signal.

## Which windows are usable

A window is usable when its **last hour** — the one the label attaches to — has at
least `--min-stations` stations present (default 1). An earlier hour being empty
is what the mask is for. In regression mode, windows past the last qualifying
event are dropped as well: their target is unknown, not long.
