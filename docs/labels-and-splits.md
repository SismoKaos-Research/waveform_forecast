# Labels, folds, and leakage

The catalogue is **only the answer key**. There is no catalogue feature anywhere
in the model's input: the inputs are the stations' waveform-derived features, and
`catalog.py` supplies the target.

By default the label is region-wide (`AEGEAN_BBOX = 36–40°N, 25–30°E`), exactly
as in `catalog_mlp`, so the two projects answer the same question.

## The assumption that region-wide label rests on

**Every station has to sit inside the labelled region.** That held while the
stations were Aegean. It stopped holding, silently, when Marmara stations were
added:

| station | lat, lon | inside `AEGEAN_BBOX`? | zone |
|---|---|:-:|---|
| MANT | 38.49, 28.56 | ✅ | Aegean extensional |
| DEMI | 39.04, 28.72 | ✅ | Aegean extensional |
| ELBA | 41.15, 28.43 | ❌ (41.15 > 40) | Marmara / NAF |
| SEMS | 40.87, 29.74 | ❌ (40.87 > 40) | Marmara / NAF |

ELBA and SEMS were being scored on earthquakes 221–295 km to their south, in a
different tectonic province, that their own seismograms have essentially no view
of. Nothing errored — the join succeeded, every label was valid, the run
completed. That is the failure mode `regions.py` exists to prevent, and
`tests/test_regions.py::test_the_marmara_stations_are_outside_the_aegean_box`
pins it so a fifth out-of-box station fails loudly instead.

This is not a bug in `catalog.py`, which is a byte-identical port and stays that
way. Region-local labelling lives in `regions.py` beside it.

## Region-local labels (`--label-radius-km`)

`regions.load_events_near` is the region-local counterpart to
`catalog.load_aegean_events`: events within `max_dist_km` of the **nearest
station in the zone**, and **no bounding box** — the box is exactly what excludes
the Marmara stations' own seismicity.

Nearest-station, not centroid: ELBA and SEMS are 114 km apart, so a centroid sits
~57 km from each and is a place no instrument is. A zone of two stations is two
overlapping disks, not one big one. `catalog.station_distance_mask` already used
the nearest-station minimum and this keeps that convention.

With the flag unset, both zones fall back to the region-wide label and behaviour
is unchanged.

### Effective sample size is reported before anything trains

The catalogue starts in 2000; the archives start in 2024. So a zone's
catalogue-wide event count is several times its usable one, and `describe_zone`
reports **the count inside the archive span**, warning below 20:

```
  train zone: ELBA+SEMS
      8 qualifying event(s) inside the archive span, within 150 km of the nearest
          (39 in the catalogue overall; the earlier ones still set `days since previous`)
      [!] 8 events is a small effective sample. Aftershock sequences
          make the independent count smaller still -- read the per-fold spread, not the mean.
```

Reporting 39 where the answer is 8 is precisely the confusion this project keeps
hitting.

## The three labels

### `--mode classify` (default) — `catalog.label_hours`

> Does an M ≥ `--threshold` (4.5) event occur within `--horizon-days` (14)
> of this hour — region-wide, or inside `--label-radius-km` of the zone?

**The horizon starts when the features END, not when the hour starts.**
`hourly_index` holds hour *starts*, and the features for hour H are aggregated
over [H, H+1h]. A horizon opening at H would count an event occurring inside the
very window the model is shown — visible in the features and labelled as future,
so the model can read off part of its own answer. It is one hour of a 720-hour
horizon, so it inflates rather than invents, but it is the same window-end
mistake the `Zaman_Dk` handling was written to avoid.

`feature_hours=0` restores the old behaviour, for reproducing a figure published
before this. Every forecasting number in the repo predates it.

The horizon is converted to **seconds**, not integer days: `timedelta64` with an
int day count silently truncated, so `--horizon-days 0.5` became a *zero*-day
horizon and every label came out negative.

### `--mode regress` — `catalog.days_to_next_major`

> How many days until the next M ≥ threshold event?

The mirror of `days_since_prev_major`, with one problem that function does not
have: **the archive ends, and the catalogue ends with it**. Every hour after the
last qualifying event has no next event at all. That is right-censoring — not a
zero, not a large number — so those hours come back NaN and `run` drops them.
Filling them would teach the model that the quietest period in the record is the
one just before the archive stops.

`--cap-days` censors the wait at a bound and returns a `censored` mask alongside.
The mask is returned rather than folded away because a target of 30.0 because the
wait was 30 days and one of 30.0 because the wait was nine months are not the same
observation, and the fraction of each belongs in the report.

**Cap only to ask a bounded question.** At M≥4.5 the median wait on this archive
is ~15 days, so a 14-day cap makes ~52% of the target the cap itself and the model
is mostly asked to predict a constant. The cap is *optional* precisely because the
split is kept honest by the per-hour purge below rather than by a fixed horizon.

`--target-transform log1p` (default) fits in `log1p(days)` and inverts with
`expm1` for reporting. The wait distribution is heavy-tailed and a plain squared
error on raw days is dominated by the longest waits. **Metrics are always reported
in days.**

### `--mode rate` — `catalog.label_hours_rate_change`

> Will the next `--horizon-days` window contain **more** events than the trailing
> `--baseline-days` one?

A rate/acceleration forecast — what ETAS-family models and CSEP evaluation
actually target. It uses a much lower magnitude threshold (`--rate-threshold`,
default 3.5), so the label is driven by ~10³ events instead of the ~10¹ that
drive `label_hours` at M≥4.5. On the Aegean pair that is 1,031 events within
100 km versus 29.

The natural baseline is strongly **anti**-correlated: during an aftershock
sequence a high trailing rate predicts a *decrease* (Omori decay). Score against
`evaluate.rate_persistence_auc`, never against 0.5.

<a id="the-rate-labels-floor-is-near-degenerate"></a>

### ⚠ The rate label's floor is near-degenerate

**Read this before choosing rate mode.** The label is
`forward_count > trailing_count`, and **the trailing count is known at prediction
time**. So the trivial baseline is handed half of the comparison for free: when
the trailing count is extreme, the answer is close to determined.

Measured on the Marmara pair, the oriented persistence floor per fold came out at

```
0.6148   0.7791   0.9905   0.9154   0.8101      (mean 0.8220)
```

A floor of 0.99 is not a hard baseline, it is an almost unbeatable one. Rate mode
genuinely solves the sample-size problem and replaces it with a bar that is
structurally near-impossible to clear.

The label remains useful as a *diagnostic* — a model at chance here has learnt
nothing, which is informative — but "beats the floor" is close to unachievable by
construction. Honest alternatives, neither implemented:

- forecast the **forward count itself** against a climatological rate, or
- compare the forward window against a **long-run** baseline rather than the
  immediately preceding one, so the baseline no longer contains the answer.

## Walk-forward cross-validation

`splits.walk_forward_splits`. Expanding-window, chronological, never shuffled.

With `--cv-folds N`, the usable window indices are cut into `N + 2` blocks, and
fold *k* is:

```
train = blocks[0 .. k]      (expanding)
val   = blocks[k+1]
test  = blocks[k+2]
```

So each fold's test block is strictly later than everything it trained on, and a
later fold's test block was an earlier fold's future. `--cv-folds 1` falls back to
a single chronological `--train-frac` / `--val-frac` split.

Block edges are equal-width by default; passing `labels` places them by
**cumulative positive mass** instead, so each block holds a comparable number of
positives rather than a comparable number of hours.

`splits.print_split_diagnostics` prints the label's movement in 10 equal-width
blocks with the dominant split marked, and warns when the test positive rate is
more than 1.5× (or less than 1/1.5×) the train rate — the sign of a swarm or a
quiet period concentrated in one split rather than a model generalizing. It
happens on this archive, and it is why the per-fold floor matters more than the
pooled mean.

## Leakage control

Two distinct problems, handled separately.

### 1. Input overlap — the `seq_hours − 1` embargo

A window ending just inside val reads hours that belong to train. Every mode
drops `seq_hours − 1` indices at each block boundary for this.

### 2. Label lookahead — different in each mode

**Classify: a constant term.** The label at hour H is decided by events in
[H, H+horizon], so without the horizon term the last ~14 days of every block carry
labels determined by events in the *next* block (Lopez de Prado, *Advances in
Financial Machine Learning*, Ch. 7). The lookahead is exactly `horizon_days` for
every hour, so a constant embargo is exact:

```
embargo = seq_hours − 1 + round(horizon_days × 24)      # 359 h at 24/14 d
```

**Regress: a per-hour purge.** The lookahead is the distance to the next event,
which is different for every hour — two days inside an aftershock sequence, 183
at the longest gap in this catalogue. Embargoing the worst case would cost a
quarter of a two-year archive to protect the handful of hours that need it.

So `catalog.label_resolution_index` computes, for each hour, **the first grid
position whose data you need in order to know that hour's label**, and
`splits.purge_by_label_span` drops the hours whose label only settles in a later
block. That is Lopez de Prado's purging by label span applied exactly, rather than
bounded by the worst case.

On the MANT+DEMI split it costs **0.9% of train**, where a 14-day embargo cost 336
hours.

Details:

- Train is purged against the start of **val**; val against the start of **test**.
  **Test is never purged**: its labels resolving after the record ends is a
  property of the record, and hours with no resolution at all were dropped
  upstream.
- A censored hour settles **at the cap**, whichever comes first — "at least 30
  days" is knowable 30 days later, without waiting for an event that may never
  come. That is the whole point of censoring.
- A label that never settles inside the record gets position `len(hours)`, so no
  split can contain it.
- `searchsorted(..., side="right")`: an event falling *inside* hour p needs hour
  p, so the answer is p+1. Off by one here silently keeps the contaminated hour.
- The hour grid need not be contiguous — `align` produces a union index with gaps
  in it, and the purge is written against positions, not against a regular clock.

## The spatial holdout (`--test-stations`)

Train on one fault zone, test on another — separated in **space and time at
once**, because the fold geometry is still walk-forward:

```bash
waveform-forecast train --features hourly_4sta.parquet \
    --stations aegean --test-stations marmara \
    --label-radius-km 150 --mode rate --rate-threshold 3.0 ...
```

Three things make it a measurement rather than a mistake.

### The architecture transfers for free

`MaskedStationPool` is attention over the station axis and `project` is one
`Linear` shared across stations, so nothing in the model is tied to a particular
station's position — or even to how many there are. A model trained on two Aegean
stations evaluates on two Marmara ones with the same weights. That is a property
of the pooling design rather than a coincidence, so
`tests/test_spatial_holdout.py` tests it as one, including permutation-invariance
over the station axis (otherwise a zone's result would depend on the order the
codes were typed).

### The held-out zone has to be standardizable

This is the subtle one. `fit_stats` fits mu/sd per `(station, feature)` on the
**training windows** — and a held-out *zone* appears in no training window at
all. Its statistics would be all-NaN, fall back to the identity `(0, 1)`, and its
features would reach the model **raw while the training stations arrived
z-scored**. The run would measure a scaling mismatch and report it as failed
transfer.

Three candidate fixes, only one of which is honest:

| fix | verdict |
|---|---|
| pool one shared statistic across all stations | ✗ destroys per-station site response, which is why these are per-station |
| fit the held-out station on its own **test-period** data | ✗ the evaluation split's distribution entering the model |
| fit every station on its own hours inside the **training time window** | ✓ |

The third is what `fit_stats(..., all_station_rows=True)` does. The Marmara pair
was recording throughout the Aegean training period — it simply was not the
labelled zone. Those hours are its own, they are in the past relative to the test
block, and they carry no label at all, so using them costs nothing.
`all_station_rows=False` restores the pre-spatial behaviour.

### Each split is judged on its own zone

- **Usable windows** are computed per zone: a window with only training stations
  up is not a usable *test* window.
- **Fold boundaries** come from the training zone, so a spatial run's blocks
  cover the same stretches of time as its in-zone counterpart and the two are
  comparable. Test indices the test zone cannot supply are dropped **without
  moving the boundary**, so the temporal separation the embargo bought is intact.
- **Coverage is reported per fold**, and a fold where the held-out zone is
  absent is skipped rather than scored — there is nothing to transfer *to*.
- **The rate floor comes from the zone being scored.** The test zone's own
  trailing seismicity is what a free rule would have had; taking it from the
  training zone would score the model against a baseline built on data it is
  being tested away from.

**Without `--label-radius-km` the whole thing is meaningless** — both zones carry
the same region-wide label, so the "held-out" zone is predicting the training
zone's earthquakes. `run` prints a warning when the flag is missing.

## Timezone

The features table carries an explicit UTC index, which is right for an artifact.
The catalogue is parsed tz-naive (`%d/%m/%Y %H:%M:%S`) and every label function
ported from `../forecast` works in naive UTC. `build_inputs` converts to UTC and
*only then* drops the marker — `tz_localize(None)` alone on a non-UTC index would
silently shift every label by the offset.
