# Measured performance

**Headline: the model does not beat its floor.** Across seven walk-forward runs on
the four-station archive it cleared its own fold's bar in 1/5, 1/5, 0/5, 0/5, 0/5,
2/5 and 0/5 folds. In every run the mean AUC sat **below the mean floor**, and the
fold-to-fold spread exceeded the margin over the floor — the condition under which
a pooled number misleads. This matches the project's prior negative result and
does not overturn it.

Three things were varied, and none of them helped:

| lever | runs | outcome |
|---|---|---|
| **more stations** (2 → 4) | 1 vs 2 | +0.021 mean AUC against a fold SD of 0.18 |
| **region-local labels + a rate target** | 4–5 | model at **chance** in both tectonic zones |
| **more features** (2 → 207) | 6–7 | no gain; on the identical rate label, *worse* |

Everything below is from runs executed on 2026-09-12. The raw console logs are in
[`runs/`](runs/) — `classify_mant_demi.log`, `classify_4sta.log`,
`regress_mant_demi.log`, `rate_marmara_inzone.log`, `rate_aegean_inzone.log`,
`rate_aegean_wide207.log`, `classify_aegean_wide207.log` — and every number here
is copied from them rather than recomputed.

## The corpus

`hourly_4sta.parquet` — 20,136 hourly rows, 2024-04-23 → 2026-08-09, 207
aggregated feature columns per station.

| station | hours present | % of grid | distinct days | span |
|---|---:|---:|---:|---|
| MANT | 17,721 | 88.0% | 743 | 2024-05-01 → 2026-08-09 |
| DEMI | 12,880 | 64.0% | 538 | 2024-09-25 → 2026-08-09 |
| SEMS | 11,985 | 59.5% | 503 | 2024-04-23 → 2025-12-22 |
| ELBA | 9,613 | 47.7% | 403 | 2024-04-23 → 2025-12-31 |

(This table is `SEMS` where the README's archive survey lists `GCAM`; the
four-station parquet on disk holds MANT/DEMI/ELBA/SEMS.)

### The masked pool earns its place

| stations used | hours with **all** present | hours with **any** present |
|---|---:|---:|
| MANT + DEMI | 10,792 (53.6%) | 19,814 (98.4%) |
| MANT + DEMI + ELBA + SEMS | **3,643 (18.1%)** | **20,136 (100.0%)** |

An intersection-only four-station model would train on 18.1% of the grid. The
masked pool trains on all of it. This is the design's core claim, and it holds:
adding stations *increases* usable data here instead of shrinking it.

Pairwise overlap, in hours: MANT–SEMS 10,962 · MANT–DEMI 10,792 · MANT–ELBA
8,695 · ELBA–SEMS 8,366 · DEMI–SEMS 5,853 · DEMI–ELBA 5,008.

### Labels

- M ≥ 4.5, Aegean bbox, 14-day horizon → **hourly positive rate 0.403**
- Days-to-next-M4.5, uncapped → **median 15.53 d, mean 33.20 d**; 2,890 hours
  (14.4%) fall past the last qualifying event (2026-04-11) and are dropped as
  right-censored rather than imputed.

## Run 1 — classify, MANT + DEMI

```
waveform-forecast train --features hourly_4sta.parquet --stations MANT DEMI \
  --catalog-path cnn_earthquake/catalogs/catalog_current.csv \
  --keep-features Z_STA_LTA_Max_max EN_CROSS_CORR_mean \
  --horizon-days 14 --cv-folds 5
```

19,809 usable windows of 20,113; embargo 359 h; seeds 42/43/44; `cuda`.

| fold | train / val / test | test pos rate | floor AUC | **ensemble AUC** | per-seed AUC | seed spread | clears floor |
|---|---|---:|---:|---:|---|---:|:-:|
| 1 | 2829 / 2471 / 2471 | 0.631 | 0.6587 | **0.8082** | 0.8081, 0.3532, 0.8109 | **0.4578** | ✅ |
| 2 | 5300 / 2471 / 2471 | 0.709 | 0.6350 | **0.4543** | 0.4151, 0.4577, 0.3854 | 0.0722 | ❌ |
| 3 | 7771 / 2471 / 2471 | 0.805 | 0.5806 | **0.3009** | 0.2939, 0.5715, 0.3184 | 0.2775 | ❌ |
| 4 | 10242 / 2471 / 2471 | 0.430 | 0.7348 | **0.5341** | 0.4963, 0.5361, 0.5511 | 0.0548 | ❌ |
| 5 | 12713 / 2471 / 2471 | **0.000** | 0.5000 | **nan** | nan, nan, nan | — | — |

```
ensemble AUC:  mean 0.5244  std 0.1840
floor AUC:     mean 0.6218  std 0.0785
beats its own fold's floor in 1/5 folds
[!] fold spread exceeds the margin over the floor
```

Fold 1 in full (the only fold that cleared its bar):

| metric | value |
|---|---:|
| accuracy | 0.7163 |
| balanced accuracy | 0.7495 |
| precision / recall / F1 | 0.8959 / 0.6231 / 0.7350 |
| ROC-AUC / PR-AUC | 0.8082 / 0.8190 |
| MCC | 0.4851 |
| Brier / log-loss | 0.2308 / 0.6594 |
| Brier skill vs persistence | +0.6379 |

and fold 2, for contrast — the model collapsed to predicting the negative class
(recall 0.0011) on a block where 71% of hours were positive:

| metric | value |
|---|---:|
| accuracy | 0.2910 |
| balanced accuracy | 0.4985 |
| precision / recall / F1 | 0.4000 / 0.0011 / 0.0023 |
| ROC-AUC | 0.4543 |
| MCC | −0.0306 |
| Brier skill vs persistence | −0.0678 |

## Run 2 — classify, all four stations

Same command with `--stations MANT DEMI ELBA SEMS`. 20,113 usable windows of
20,113 (every window has at least one station).

| fold | test pos rate | floor AUC | **ensemble AUC** | seed spread | clears floor |
|---|---:|---:|---:|---:|:-:|
| 1 | 0.595 | 0.6110 | **0.8271** | 0.3691 | ✅ |
| 2 | 0.714 | 0.6365 | **0.4479** | 0.0195 | ❌ |
| 3 | 0.843 | 0.5745 | **0.3702** | 0.2035 | ❌ |
| 4 | 0.422 | 0.7120 | **0.5369** | 0.0589 | ❌ |
| 5 | **0.000** | 0.5000 | **nan** | — | — |

```
ensemble AUC:  mean 0.5455  std 0.1730
floor AUC:     mean 0.6068  std 0.0699
beats its own fold's floor in 1/5 folds
```

**Two stations vs four changes nothing that survives the spread.** Mean AUC moves
0.5244 → 0.5455 (+0.021) against a fold SD of ~0.18, and the per-fold pattern is
identical: the same fold clears, the same three miss, the same one is
unmeasurable. Adding ELBA and SEMS bought 100% window coverage and no skill.

The per-fold station coverage explains why less happened than the station count
suggests — test-block coverage per station, from the four-station run:

| fold | MANT | DEMI | ELBA | SEMS |
|---|---:|---:|---:|---:|
| 3 | 80.0% | 63.5% | 80.0% | 80.4% |
| 4 | 59.9% | 80.0% | **3.3%** | **0.0%** |
| 5 | 100.0% | 99.6% | **0.0%** | **0.0%** |

ELBA's and SEMS's archives end in December 2025, so the two latest folds — the
ones with the most training data behind them — are MANT+DEMI runs under a
four-station name. This is exactly the diagnostic `run_fold` prints for, and it
is the reason the four-station numbers should not be read as a four-station
result.

## Run 3 — regress, MANT + DEMI

```
waveform-forecast train --mode regress --features hourly_4sta.parquet \
  --stations MANT DEMI --catalog-path .../catalog_current.csv \
  --keep-features Z_STA_LTA_Max_max EN_CROSS_CORR_mean --cv-folds 5
```

Uncapped target, `log1p` transform, per-hour label-span purge, embargo 23 h.
16,919 usable windows of 20,113 after dropping the 2,890 right-censored ones.

| fold | constant MAE | persistence MAE | floor | **ensemble MAE** | skill | model ρ | persistence ρ | pred_std | beats floor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| 1 | 16.269 | 23.143 | 16.269 | **19.230** | −0.182 | −0.2418 | −0.4752 | 4.16 | ❌ |
| 2 | 6.231 | 9.869 | 6.231 | **6.807** | −0.093 | +0.0617 | +0.3546 | 2.18 | ❌ |
| 3 | 29.592 | 11.506 | 11.506 | **37.529** | −2.262 | −0.0126 | −0.1037 | 16.71 | ❌ |
| 4 | 13.782 | 10.742 | 10.742 | **22.619** | −1.106 | −0.3808 | +0.0175 | 13.35 | ❌ |
| 5 | 9.813 | 23.747 | 9.813 | **10.443** | −0.064 | −0.1098 | +0.2328 | 3.14 | ❌ |

```
ensemble MAE:  mean 19.325  std 10.750 d
floor MAE:     mean 10.912  std  3.231 d
beats its own fold's floor in 0/5 folds
[!] persistence orders the hours better in 3/5 fold(s)
[!] fold spread exceeds the margin over the floor
```

This is the cleaner negative of the three. The model is worse than a trivial rule
on **every** fold, by 6% to 226%, and its rank correlation is negative on four of
five — it is not merely uninformative, it orders the hours slightly backwards.
Note fold 3 and 4, where `pred_std` (16.7 and 13.3 days) shows the model was *not*
collapsed to a constant: it produced a confident, varied, wrong ordering.

The purge cost varies enormously by fold, which is the point of doing it per hour
rather than by a fixed embargo:

| fold | train hours purged | % of train | val hours purged |
|---|---:|---:|---:|
| 1 | 564 | 23.3% | 2,394 (**the entire val block**) |
| 2 | 2,958 | 61.5% | 348 |
| 3 | 348 | 4.8% | 73 |
| 4 | 73 | 0.8% | 0 |
| 5 | 0 | 0.0% | 494 |

The 0.9%-of-train figure quoted in `train.py`'s docstring is achievable (fold 4),
but it is not typical: on the boundaries that fall inside a long catalogue gap the
purge is expensive, and on fold 1 it removed the entire validation block. That
fold trained with no validation split at all.

## Runs 4 and 5 — rate mode, region-local labels, each zone in-zone

The first runs under `regions.py`: each zone labelled by **its own** events
rather than the region-wide `AEGEAN_BBOX`, which for the Marmara pair was
previously earthquakes 221–295 km away in a different province.

```
waveform-forecast train --features hourly_4sta.parquet \
  --stations {marmara|aegean} --catalog-path .../catalog_current.csv \
  --mode rate --rate-threshold 3.0 --label-radius-km 150 \
  --keep-features Z_STA_LTA_Max_max EN_CROSS_CORR_mean \
  --horizon-days 14 --cv-folds 5
```

| zone | stations | M≥4.5 in span | M≥3.0 in span (the rate label) | positive rate |
|---|---|---:|---:|---:|
| Marmara | ELBA+SEMS | **8** | 146 | 0.364 |
| Aegean | MANT+DEMI | 30 | **1,084** | 0.427 |

| fold | Marmara floor | **Marmara AUC** | Aegean floor | **Aegean AUC** |
|---|---:|---:|---:|---:|
| 1 | 0.6148 | **0.4502** | 0.5137 | **0.3125** |
| 2 | 0.7791 | **0.6656** | 0.8416 | **0.5384** |
| 3 | 0.9905 | **0.3932** | 0.8707 | **0.4346** |
| 4 | 0.9154 | **0.5233** | 0.7375 | **0.5450** |
| 5 | 0.8101 | **0.5205** | 0.8294 | **0.5743** |

```
Marmara   ensemble AUC: mean 0.5106  std 0.0913   floor mean 0.8220   0/5 folds
Aegean    ensemble AUC: mean 0.4809  std 0.0966   floor mean 0.7586   0/5 folds
```

**The model is at chance in both zones.** 0.5106 and 0.4809, five folds and three
seeds each. That is a stronger statement than losing to a hard baseline: the two
waveform features carry essentially no information about seismicity rate.

**The Aegean run is the control that rules out "Marmara is simply too quiet."**
It has **7.4× the events** driving its label (1,084 vs 146) and does no better —
marginally worse, in fact. So the negative is not a sample-size artefact of the
Marmara zone; it is a property of these features.

### Two caveats on these two runs specifically

**The rate floor is near-degenerate, and that is a flaw in the label.** The label
is `forward_count > trailing_count` and the trailing count is *known at prediction
time*, so the trivial rule gets half the comparison for free. Fold 3's floor of
0.9905 is not a hard baseline, it is an almost unbeatable one. Rate mode solves
the sample-size problem and replaces it with a structurally near-impossible bar —
see [labels-and-splits.md](labels-and-splits.md#the-rate-labels-floor-is-near-degenerate).
The *model-at-chance* finding stands regardless, because it does not depend on
where the floor sits.

**Only 2 of 207 available feature columns were used.** Both runs use
`Z_STA_LTA_Max_max` and `EN_CROSS_CORR_mean`. "These two features carry no rate
information" is what was measured; "the waveforms carry none" is not.

## Runs 6 and 7 — the wide-feature test, all 207 columns

The one lever untouched by runs 1–5: every run above used **2 of 207** available
feature columns (`Z_STA_LTA_Max_max`, `EN_CROSS_CORR_mean`). These drop
`--keep-features` entirely — 69 base features × {mean, std, max} per station,
covering the STA/LTA, Hjorth, spectral, entropy, correlation-dimension and
Lyapunov families.

Both on the Aegean pair with region-local labels (`--label-radius-km 150`).

### Run 6 — rate, 207 features

The clean comparison: **identical label, identical folds, identical floors** as
Run 5. Only the feature count changes.

| fold | floor | Run 5 (2 feat) | **Run 6 (207 feat)** |
|---|---:|---:|---:|
| 1 | 0.5137 | 0.3125 | **0.3087** |
| 2 | 0.8416 | 0.5384 | **0.5145** |
| 3 | 0.8707 | 0.4346 | **0.3363** |
| 4 | 0.7375 | 0.5450 | **0.5415** |
| 5 | 0.8294 | 0.5743 | **0.5170** |

```
2 features:    mean 0.4809  std 0.0966   floor mean 0.7586   0/5 folds
207 features:  mean 0.4436  std 0.0997   floor mean 0.7586   0/5 folds
```

**100× the features moved the model from 0.4809 to 0.4436 — slightly worse, and
worse on four of five folds.** The floors are identical to four decimal places,
confirming the two runs differ in nothing but the input width. This is the
strongest single piece of evidence in the whole set: the feature count was not
what was limiting anything.

### Run 7 — classify, 207 features

| fold | floor | **ensemble AUC** | per-seed spread | clears |
|---|---:|---:|---:|:-:|
| 1 | 0.5027 | **0.5396** | 0.1468 | ✅ |
| 2 | 0.5279 | **0.5959** | 0.0377 | ✅ |
| 3 | 0.6176 | **0.5285** | 0.0796 | ❌ |
| 4 | 0.7348 | **0.4416** | 0.0522 | ❌ |
| 5 | 0.5000 | **nan** | — | — |

```
ensemble AUC:  mean 0.5264  std 0.0552
floor AUC:     mean 0.5766  std 0.0899
beats its own fold's floor in 2/5 folds
[!] fold spread exceeds the margin over the floor
```

2/5 is the best fold count in the set, and it is still **not a result**:

- The **mean is below the mean floor** (0.5264 vs 0.5766).
- The two clearing margins are **+0.037 and +0.068** against a fold SD of 0.055 —
  inside the noise.
- **The floors also fell** (0.50–0.53 on folds 1–2, against 0.62–0.66 in Run 1).
  Region-local labelling weakened persistence, so part of "clearing the floor" is
  the bar moving down rather than the model moving up.
- **Fold 1 again had no checkpoint selection** — all-negative val block, `val AUC
  nan` every epoch. Its 0.5396 is an unselected 8-epoch model.
- Folds 3 and 4 regressed as soon as the floor rose, which is what a model
  tracking the base rate rather than the signal does.

### It overfits, exactly as the code warned

`train.py` says `--keep-features` is "usually not optional" because the table is
~200 columns per station while the positive class is a handful of events per
fold. Run 7's first fold, seed 42:

```
epoch 1  train loss 0.7020   val loss 0.3518
epoch 3  train loss 0.0209   val loss 0.2918
epoch 8  train loss 0.0098   val loss 0.5525
                             -> test loss 7.4531   test AUC 0.3701
```

and on fold 2 the val loss climbs past **14** while train loss reaches 0.05. With
207 features × 2 stations and 30 qualifying events in span, the model memorises
the training block. The warning in the docstring was correct for classify — but
note it did **not** apply to Run 6, where the rate label is driven by 1,084
events and the model still gained nothing.

## Caveats that change how these numbers read

Three, all visible in the logs, none of which the summary line captures:

**1. Fold 5's test block has zero positives.** Its positive rate is 0.000, so AUC
is undefined, and `safe_auc` correctly returns NaN rather than a wrong number. In
both classification runs the reported "5/5 folds" is really 4 measured folds and
one unmeasurable one, and `mean 0.5244` is a mean over four.

**2. Fold 1 had no checkpoint selection at all.** Its validation block is also
all-negative, so `val AUC` was NaN on every epoch of every seed (24 epochs total
across 3 seeds = exactly `--patience 8` each). No epoch registered as an
improvement, `best_state` stayed `None`, and the scored model is whatever the
weights were after 8 epochs. **The only fold that beat its floor in either
classification run is the one where the model was never selected** — that number
should be read as a lucky stopping point, not as a result.

**3. Per-seed spread reaches 0.46 AUC.** Fold 1 of run 1: seeds 42/43/44 scored
0.8081, 0.3532, 0.8109. The ensemble of those three is 0.8082. A single-seed run
of this configuration could have reported anything from "well below chance" to
"comfortably above the floor" from identical data and identical code. This is why
`fold_result` prints the spread and why `summarise` refuses a bare AUC.

## What this does and does not show

**Does:**

- The pipeline runs end to end on a real four-station archive, on GPU, and its
  diagnostics fire correctly on every degenerate case it encounters — NaN AUC on
  a single-class block, a station absent for an entire training split, a fully
  purged validation block, test/train skew, spread-exceeds-margin.
- The masked pool delivers on the coverage argument: 18.1% → 100.0% of the grid
  usable at four stations.
- The evaluation refuses to flatter the model. Every negative above is reported by
  the code itself, unprompted.

**Does not:**

- Show that multi-station waveform features forecast M≥4.5 events at a 14-day
  horizon. They did not clear the floor here.
- Show that four stations beat two. The difference (+0.021 mean AUC) is a
  ninth of the fold SD, and the two folds with the most training data behind them
  had ELBA and SEMS at ~0% coverage.
- Test `--arm raw`. These runs are all `--arm features`; the raw CNN arm needs the
  5 Hz waveform archive, which is not what `hourly_4sta.parquet` holds.
- Tune anything. Defaults throughout. Runs 1–5 use two hand-picked feature
  columns; runs 6–7 use all 207 and do no better.
- Test the **spatial holdout**. `--test-stations` is implemented and tested
  (`tests/test_spatial_holdout.py`), but running it would be premature: there is
  no in-zone signal in either zone to transfer.

## Reproducing

The three commands are quoted verbatim above. Runtime on one GPU was roughly 10–25
minutes per 5-fold run at these settings. The test suite —

```
$ uv run pytest
106 passed in 3.61s
```

— covers the properties these results depend on: that an absent station
contributes exactly nothing, that statistics are fitted on train only, that the
purge drops what it claims to, and that a bare AUC cannot be summarised.
