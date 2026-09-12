# Architecture

One forecast per window, from however many stations happened to be recording.

```
     x : (batch, hours, stations, features)         present : (batch, hours, stations)
                    │                                            │
       ┌────────────▼────────────┐                                │
       │ encoder (arm B only)    │  RawWaveformEncoder, 1D CNN    │
       │ (b·t·s, 3, N) → (·, 32) │  over one hour of 5 Hz samples │
       └────────────┬────────────┘                                │
                    │  arm A passes the feature vector through    │
       ┌────────────▼────────────┐                                │
       │ project                 │  Linear(feat_dim, dim) → GELU  │
       │                         │  → Dropout                     │
       └────────────┬────────────┘                                │
       ┌────────────▼────────────────────────────────────────────▼┐
       │ MaskedStationPool       │  zero masked cells, score,     │
       │ (batch, hours, dim)     │  −inf-mask, softmax, weighted  │
       │                         │  mean over the STATION axis    │
       └────────────┬────────────┘                                 
       ┌────────────▼────────────┐
       │ LSTMAttentionBranch     │  BiLSTM(dim→hidden) → MHSA(4)
       │ (batch, hidden*2)       │  → LayerNorm(h + attn) → mean over TIME
       └────────────┬────────────┘
       ┌────────────▼────────────┐
       │ head                    │  LayerNorm → Dropout → Linear(→hidden)
       │ (batch,)                │  → GELU → Dropout → Linear(→1)
       └─────────────────────────┘
                    │
              one raw logit (classify)  /  one value in log1p-days (regress)
```

Two axes are collapsed, in this order and not the other: **stations first**
(masked attention pool), **then hours** (LSTM + self-attention, mean over time).
Pooling stations first is what lets an hour with one station up and an hour with
four both become a single `dim`-wide vector the recurrent branch can read as one
sequence.

## The station pool is the whole design

`model.MaskedStationPool`.

The stations do not share a span. Measured from the AFAD archives on disk:

| station | days | span |
|---|---|---|
| MANT | 756 | 2024-05-01 → 2026-08-18 |
| DEMI | 563 | 2024-09-25 → 2026-08-18 |
| ELBA | 252 | 2024-04-23 → 2025-05-05 |
| GCAM | 189 | 2024-05-01 → 2024-12-17 |

DEMI+MANT overlap on 479 days; all three Aegean stations on **80**; all four on
29. A model that concatenates station vectors requires every station present, so
it would train on those 80 days — of order two qualifying events, the
effective-sample-size trap this project has hit before. Pooling over a presence
mask lets it train on the **union** instead, using whichever stations are up.

### An absent station contributes exactly nothing

Not "approximately nothing". With one station present the pooled embedding *is*
that station's embedding, to the bit, whatever garbage sits in the other slots.
Three things make that true, and each was a bug first:

1. **`features` writes NaN for an absent hour, never 0.** Zero after
   standardization means "exactly average" — a reading the model cannot
   distinguish from a real quiet one.
2. **The pool masks before the softmax, not after.** Zeroing weights afterwards
   still lets an absent station take probability mass and shrink the pooled
   vector toward the origin.
3. **The pool zeroes masked cells before weighting them.** `NaN × 0` is NaN, so a
   zero weight alone does not neutralise an absent cell: it poisons the pooled
   vector, then the loss, then every gradient after it.

An hour with *no* station present would be all `-inf` before the softmax, and
softmax of that is NaN. The pool substitutes a uniform row and then zeroes the
result; `any_present` tells the caller the hour was empty rather than quiet.

This is tested as a property, not asserted — see `tests/test_model.py`:

- `test_one_station_present_pools_to_that_station_exactly`
- `test_garbage_in_an_absent_slot_changes_nothing`
- `test_absent_stations_take_no_weight`
- `test_an_empty_hour_does_not_poison_its_neighbours`
- `test_gradients_stay_finite_through_a_masked_pool`

### Why a learned projection before pooling

`proj_dim` (default `feat_dim`) projects each station into a common space first.
Site response is a property of the station, not the earthquake: two stations
looking at the same event report different amplitudes, and averaging raw
per-station vectors would pool quantities that are not on the same scale.

### Which station did it lean on

`MultiStationForecaster.last_weights` keeps the pooling weights from the last
forward pass, so a run can report which station the model actually used rather
than assert that multi-station helped.

## The recurrent branch

`blocks.LSTMAttentionBranch` — unchanged from
`cnn_earthquake/src/sismokaos/model/blocks.py`. Bidirectional LSTM for long-range
order, multi-head self-attention to weight the steps, a residual + LayerNorm, and
a mean over time. Output width is `hidden * 2`.

Keeping it byte-identical is what lets a multi-station number be read against the
published single-station figures: the difference between the runs is the station
axis, not the sequence model.

<a id="the-two-arms"></a>

## The two arms

They differ by **one constructor argument**.

| | `--arm features` | `--arm raw` |
|---|---|---|
| `encoder` | `None` | `RawWaveformEncoder(out_dim=proj_dim or 32)` |
| input shape | `(b, t, s, feat_dim)` | `(b, t, s, 3, samples)` |
| what the pool sees | the hourly feature vector, already an embedding | a 1D CNN embedding of one hour of 5 Hz 3-component samples |

`RawWaveformEncoder` is four `Conv1d → BatchNorm → GELU → Dropout` blocks with
stride 4 each, then adaptive average pooling — unchanged from
`cnn_earthquake/.../waveform.py`, the shape the published single-station runs
used.

**`--arm raw` could not run until `waveforms.py` existed.** The flag, the
encoder and the `--help` text shipped in the first commit, but `build_inputs`
only ever read the hourly *feature* parquet — so the flag handed a 3,072-wide
aggregate vector to a `Conv1d` expecting three channels of samples:

```
RuntimeError: Given groups=1, weight of size [16, 3, 7], expected
input[1, 3072, 1] to have 3 channels, but got 3072 channels instead
```

`waveform-forecast waveforms` builds the missing tensor; see
[data-pipeline.md](data-pipeline.md#the-raw-tensor). The raw arm needs a much
smaller `--seq-hours` and `--batch-size` than the feature arm: at 24 h and batch
64 one batch is 664 MB, against 55 MB at 8 h and batch 16.

Splits, purge, floor and folds are shared between the arms. That sharing is what
makes them a comparison rather than two unrelated runs.

## Training loop

`train.train_one_seed`, per seed, per fold:

| | classify | regress |
|---|---|---|
| criterion | `BCEWithLogitsLoss(pos_weight=(1−p)/p)` from the **train** positive rate | `HuberLoss(delta=1.0)` in the fitted space |
| optimiser | `AdamW(lr=3e-4, weight_decay=0.1)`, cosine-annealed over `--epochs` | same |
| grad clip | `clip_grad_norm_(…, 1.0)` | same |
| checkpoint selected on | best **val AUC** | best **val MAE in days** (not the transformed loss) |
| early stop | `--patience` epochs without improvement (default 8) | same |
| reported per epoch | train loss, val loss, val AUC | train loss, val loss, val MAE (d), val Spearman |

Three seeds by default (`--ensemble-seeds 42,43,44`). The reported score is the
**mean of the seeds' probabilities/predictions**, and the per-seed spread is
printed beside it — per-seed AUC spread reaches 0.17 on this data, so a single
seed is not a measurement.

Details that are load-bearing:

- **Huber, not MSE**, in regression: even on log1p-days the tail is long, and a
  squared error lets the handful of longest waits set the gradient for every
  batch they land in. The classic outcome is a model that predicts the mean
  everywhere with a respectable MAE and zero rank correlation —
  `regression_report`'s `pred_std` exists to catch exactly that.
- **Val loss uses the same criterion the optimiser saw**, `pos_weight` included.
  An unweighted val BCE is on a different scale from the training objective and
  cannot be read against it, which is most of what a val loss is for.
- **Losses are weighted by batch size**, not averaged over batches: the last
  batch is usually short, and a plain mean of batch means lets its handful of
  windows count as much as a full one.
- **`best` starts at `-inf`, not `-1`**: in regression the tracked metric is a
  negative MAE, and `-1.0` would silently reject every checkpoint worse than one
  day.
- **`log1p`, not `log`**: the wait is zero at the instant of an event and
  `log(0)` is `-inf`. Predictions are clipped to `[0, 50]` before `expm1`,
  because an unconstrained head early in training emits large negatives and
  `expm1` of those is a physically impossible −1 day.
- **`loss.detach()` before accumulating**: reading the scalar off the live tensor
  keeps the graph alive for a whole epoch's batches — a slow memory leak on the
  raw arm, where one window is 3 × 18,000 samples.

## Known behaviour worth flagging

When a fold's **validation block contains a single class**, `safe_auc` returns
NaN, no epoch ever registers as an improvement, and `best_state` stays `None` —
so the scored model is whatever the weights were when patience ran out, with no
checkpoint selection at all. This is visible in the log as `val AUC nan` on every
epoch. It is not silent, but it does mean such a fold's number is a
fixed-epoch-count model rather than a selected one. It happened on fold 1 of the
run in [performance.md](performance.md).
