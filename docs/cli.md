# CLI reference

```
waveform-forecast                     # the command listing
waveform-forecast features --help
waveform-forecast train --help
```

`cli.main` dispatches to the underlying module's `main()` with the arguments
passed through untouched, so a recorded command and the real one cannot diverge.
Every command also runs standalone as `python -m waveform_forecast.features`.

---

## `waveform-forecast features`

Aggregate one or more stations' extractor output into a single aligned hourly
table. See [data-pipeline.md](data-pipeline.md) for what each step does.

| flag | default | meaning |
|---|---|---|
| `--station CODE=PATH` | *required*, repeatable | e.g. `MANT=feats/MANT/MANT_features.parquet`. `.parquet` or `.csv` by extension. The CODE is what `--stations` selects by at train time. |
| `--agg` | `mean,std,max` | comma-separated statistics per feature. **`max` matters**: STA/LTA and PEAK measure transients, and their hourly mean reports the noise floor around one. |
| `--min-windows` | `8` | hours built from fewer extractor windows are marked absent rather than kept as a thin reading (a full hour is ~72 windows at `step_sec` 50) |
| `--drop-dev` | off | drop the extractors' `_DEV` (inter-window diff) columns, halving the width |
| `--out` | *required* | output parquet path |

Output: hour-indexed parquet with `<STATION>__<FEATURE>_<agg>` columns and one
`present_<STATION>` boolean per station.

---

## `waveform-forecast train`

Train the multi-station forecaster and score it against its floor.

### Inputs

| flag | default | meaning |
|---|---|---|
| `--features` | *required* | the parquet `waveform-forecast features` wrote |
| `--stations A B ...` | *required* | station codes to use, as named in that table |
| `--catalog-path` | *required* | catalogue CSV — **the label, not an input** |

### What is being asked

| flag | default | meaning |
|---|---|---|
| `--arm {features,raw}` | `features` | `features`: pool the hourly vectors. `raw`: a 1D CNN over one hour of 5 Hz samples per station first. |
| `--mode {classify,regress,rate}` | `classify` | `classify`: does an M≥threshold event occur within the horizon (`catalog_mlp`'s label, unchanged). `regress`: how many days until the next one. `rate`: will the next window hold **more** events than the trailing one — driven by ~10³ events at M≥3.0 instead of ~10¹ at M≥4.5. **Read the caveat in [labels-and-splits.md](labels-and-splits.md#the-rate-labels-floor-is-near-degenerate) before using it.** |
| `--threshold` | `4.5` | magnitude defining a qualifying event |
| `--horizon-days` | `14.0` | classify only. Fractional days are honoured. |
| `--cap-days` | `None` | regress only. Censor the target at this many days. **Optional** — the split is kept honest by the per-hour label-span purge, not by a fixed horizon. At M≥4.5 the median wait is ~15 d, so a 14 d cap makes ~52% of the target the cap itself. |
| `--target-transform {log1p,none}` | `log1p` | regress only. Metrics are reported in **days** either way. |

### Geography — where the label comes from, and what is held out

The stations on disk are **two tectonic settings, not one**: MANT+DEMI are
Aegean (63 km apart), ELBA+SEMS are Marmara/North Anatolian Fault (114 km
apart), and the pairs are 221–295 km from each other. ELBA and SEMS are north of
40°N, i.e. **outside `catalog.AEGEAN_BBOX`** — under the region-wide label they
forecast a province their own seismograms cannot see.

| flag | default | meaning |
|---|---|---|
| `--label-radius-km` | `None` | label each zone with events within this many km of the **nearest station forecasting them**, instead of the region-wide `AEGEAN_BBOX`. Unset reproduces the previous behaviour exactly. **Required for a spatial holdout to mean anything** — without it both zones carry the same label and the "held-out" zone is predicting the training zone's earthquakes. |
| `--test-stations` | `= --stations` | hold these out as the test zone, e.g. `--stations aegean --test-stations marmara`. Zone names expand to codes. |
| `--station-table` | `None` | AFAD `istasyon_katalog.csv` for coordinates. Omit to use the built-in coordinates for MANT, DEMI, ELBA, SEMS. |

Zone names accepted anywhere a station code is: `aegean` → MANT DEMI,
`marmara` → ELBA SEMS. Codes and zone names mix freely.

### Rate mode

| flag | default | meaning |
|---|---|---|
| `--rate-threshold` | `3.5` | magnitude defining the events whose **rate** is forecast. Much lower than `--threshold` on purpose: the point is a label driven by many events. |
| `--baseline-days` | `= --horizon-days` | trailing window the forward window is compared against. Like-for-like by default, so the label has no window-length bias. |

### Windowing

| flag | default | meaning |
|---|---|---|
| `--seq-hours` | `24` | hours per window; the label attaches to the window's last hour |
| `--keep-features NAME ...` | all | restrict to these aggregated feature names, e.g. `Z_STA_LTA_Max_max`. The table is ~200 columns per station and the positive class is a handful of events per fold, so **this is usually not optional**. |
| `--min-stations` | `1` | drop windows whose last hour has fewer than this many stations present |

### Model

| flag | default |
|---|---|
| `--hidden` | `64` — LSTM hidden size per direction, and head width |
| `--proj-dim` | `None` (→ `feat_dim`) — width each station is projected into before pooling; also the raw encoder's `out_dim` |
| `--dropout` | `0.3` |

### Training

| flag | default |
|---|---|
| `--epochs` | `40` |
| `--batch-size` | `64` |
| `--lr` | `3e-4` |
| `--weight-decay` | `0.1` |
| `--patience` | `8` epochs without val improvement |
| `--ensemble-seeds` | `42,43,44` |
| `--num-workers` | `2` |

### Evaluation

| flag | default |
|---|---|
| `--cv-folds` | `5` — walk-forward; `1` falls back to a single chronological split |
| `--train-frac` | `0.70` — `--cv-folds 1` only |
| `--val-frac` | `0.15` — `--cv-folds 1` only |

Device is chosen automatically (`cuda` when available).

---

## Errors it refuses to proceed past

Each of these is a failure that would otherwise be quiet:

| message | cause |
|---|---|
| `has no Zaman_Dk column` | not extractor output; the rows cannot be placed on a clock |
| `has a different feature set than the first station` | stations extracted with different configs — not comparable |
| `--station wants CODE=PATH` | malformed station spec |
| `is not indexed by hour` | the features parquet was not written by `waveform-forecast features` |
| `[stations] not in this table` | a `--stations` code with no `present_` column; lists what is there |
| `--keep-features got [...]` | an unknown feature name, rather than silently selecting nothing |
| `no M>=X event in <catalog> inside the Aegean box` | nothing to label with |
| `no window has >= N station(s) at its last hour` | `--min-stations` excluded everything |
| `not enough data for a meaningful split` | a fold with <10 train or <5 test windows; that fold is skipped |
