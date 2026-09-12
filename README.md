# waveform-forecast

Forecasting whether an M≥4.5 earthquake occurs in the Aegean within the next
14 days — from several stations' continuous waveforms, against the same label
and the same floor as the catalogue-only model in `../forecast`.

```bash
# 1. aggregate the extractors' output onto one hourly clock
waveform-forecast features \
    --station MANT=feats/mant.parquet --station DEMI=feats/demi.parquet \
    --drop-dev --out hourly.parquet

# 2. train, and score against the floor
waveform-forecast train --features hourly.parquet --stations MANT DEMI \
    --catalog-path ../cnn_earthquake/catalogs/catalog_current.csv \
    --keep-features Z_STA_LTA_Max_max EN_CROSS_CORR_mean \
    --horizon-days 14 --cv-folds 5
```

## The label is `catalog_mlp`'s, unchanged

`catalog.py`, `splits.py`, `metrics.py`, `evaluate.py` and `seeding.py` are
byte-identical ports from `../forecast`. That is the point: "the same
labelling" is only a meaningful claim if the two projects share the code that
produces it. `catalog_mlp` answers *does an M≥threshold event occur within the
horizon* from catalogue features; this answers the same question from
waveforms, so the two numbers can be read against each other.

The catalogue here is **only** the answer key. It is deliberately not restricted
to each station's neighbourhood — the label is region-wide, exactly as it is
there, or the question changes.

## Features come from the extractors, not from here

`sismokaos-cli` (Rust) and `Sismokaos-featureExtract` (Python) compute them, and
agree on a column scheme: `Pencere_ID`, `Zaman_Dk`, then `{E,N,Z}_STA_LTA_Max`,
`{E,N,Z}_HJORTH_ACTIVITY`, `EN_CROSS_CORR` and the rest, each with a `_DEV`
column holding its diff against the previous window. Defining them a second time
here would give two versions of the same quantity and no way to say which
produced a number.

`features` does the part neither extractor does: collapse ~72 windows an hour
into one row (`mean`, `std`, **`max`** — STA/LTA and PEAK measure transients,
and their hourly mean is the noise floor around one), put every station on a
common index, and record who was actually there.

## Stations are pooled over a presence mask

Measured from the AFAD archives on disk:

| station | days | span |
|---|---|---|
| MANT | 756 | 2024-05-01 → 2026-08-18 |
| DEMI | 563 | 2024-09-25 → 2026-08-18 |
| ELBA | 252 | 2024-04-23 → 2025-05-05 |
| GCAM | 189 | 2024-05-01 → 2024-12-17 |

DEMI+MANT overlap on 479 days, **all three Aegean stations on 80**, all four on
29. A model requiring every station present would train on those 80 days — of
order two qualifying events, the effective-sample-size trap this project has hit
before. Pooling over a mask lets it train on the union instead.

**An absent station contributes exactly nothing.** Not "approximately nothing":
with one station up the pooled embedding *is* that station's, to the bit, with
arbitrary garbage in the other slots. Three things make that true, and each was
a bug first:

- `features` writes NaN for an absent hour, never 0 — zero after
  standardization means "exactly average", a reading the model cannot
  distinguish from a real quiet one.
- the pool masks **before** the softmax, not after — zeroing afterwards still
  lets an absent station take probability mass and shrink the pooled vector.
- the pool zeroes masked cells before weighting them — `NaN × 0` is NaN, so a
  zero weight alone does not neutralise an absent cell; it poisons the pooled
  vector, the loss, and every gradient after it.

ELBA is 235–395 km from the other three, in the Marmara region. It is a second
seismic setting rather than a fourth view of the same events, and has no near
pair on disk.

## Two arms, one argument apart

`--arm features` pools the hourly vectors directly. `--arm raw` puts a 1D CNN
(`RawWaveformEncoder`, 5 Hz, the shape the published single-station runs used)
in front of each station-hour first. Splits, purge, floor and folds are shared —
that sharing is what makes them a comparison rather than two runs.

## Every number comes with its floor

`evaluate.fold_result` computes the model's AUC, the floor it had to clear *on
that fold*, and the per-seed spread together; `summarise` raises `TypeError` on
a bare AUC. It also warns when fold spread exceeds the margin over the floor —
the condition under which a pooled number misleads.

Labels are purged at fold boundaries by `seq_hours − 1 + horizon`: the label at
hour H is decided by events in [H, H+horizon], so without the horizon term the
last ~14 days of every block carry labels determined by events in the *next*
block (Lopez de Prado, *Advances in Financial Machine Learning*, Ch. 7).

**Read this expecting a negative.** On the corrected catalogue, neither
raw-waveform CNNs nor hand-crafted continuous features beat persistence in this
project — 0 of 10 chaos sweep cells, all three sequence architectures below
floor. Multi-station is a real difference from those runs. The floor is the same
floor.

```bash
uv run pytest
```

## Documentation

`docs/` holds the full write-up: [architecture](docs/architecture.md), the
[data pipeline](docs/data-pipeline.md),
[labels and splits](docs/labels-and-splits.md),
[evaluation](docs/evaluation.md), a [CLI reference](docs/cli.md), and
**[measured performance](docs/performance.md)** — three 5-fold walk-forward runs
on the four-station archive, per fold, against the floor, with the raw logs in
`docs/runs/`. Short version: 1/5, 1/5 and 0/5 folds cleared their own floor.
