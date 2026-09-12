# Evaluation: no number without its floor

The central methodological finding of this project's forecasting work is that a
pooled number misleads on this data:

- *"Read the block-level numbers, not the pooled ones"*
- fold SD of 0.07–0.16 dwarfs every gap between models
- `feature_lstm` "beats its own fold's floor in 2 of 5 folds" while losing to
  persistence on average

**A model's AUC alone is therefore not a result.** It is a result only beside the
floor it had to clear *on that fold*, and beside the spread across folds. So
`evaluate.fold_result` computes all three at once, there is no path that produces
one without the others, and `summarise` refuses to print a summary assembled any
other way:

```python
>>> summarise([0.81, 0.45], n_folds=2)
TypeError: summarise() takes FoldResult objects, which carry the floor their
AUC was measured against. A bare AUC has no meaning here: fold SD on this data
is 0.07-0.16, and models have beaten the pooled number while losing to
persistence on most folds.
```

`FoldResult` and `RegressionFoldResult` are frozen dataclasses constructed only
by `fold_result` / `regression_fold_result`.

## Classification floors

Two trivial rules, computed on **that fold's** test block, from training data
only. The floor is `max(0.5, base_rate_auc, oriented_persistence_auc)`.

| floor | rule |
|---|---|
| base rate | predict the training majority class everywhere |
| persistence | "a qualifying event happened within one horizon already", i.e. `days_since_prev ≤ horizon_days` |

### The orientation bug this encodes

**An anti-predictive rule is exactly as exploitable as a predictive one — you
invert it.** So a baseline's achievable score is `max(auc, 1 − auc)`.

Rate mode always did this inside `rate_persistence_auc`; event mode did not, which
silently collapsed the floor to chance whenever persistence landed below 0.5. That
is what made an n=4 event-mode result look like it cleared a 0.5000 floor when the
properly oriented bar was ~0.58.

A **trained model** gets no such courtesy: `safe_auc(..., oriented=False)` for the
model, `oriented=True` for baselines. Scoring below chance with a trained model is
a failure to surface, not a sign to flip.

### What a fold prints

```
--- Floors (test set) ---
  base-rate (majority)   AUC 0.5000   n=2471
  persistence            AUC 0.3413   Brier 0.4241   n=2471
  -> floor to clear      AUC 0.6587   (oriented: a rule below 0.5 is inverted for free)

--- multistation[features] ---
  per-seed AUC: ['0.8081', '0.8074', '0.8090']  mean 0.8082  spread 0.0016
  ENSEMBLE (mean of 3 seeds' probabilities)   AUC 0.8082   n=2471
```

followed by the full `binary_report`: accuracy, balanced accuracy, precision,
recall, F1, ROC-AUC, PR-AUC, MCC, Brier, log-loss, the confusion matrix, and
`brier_skill_score_vs_persistence`.

Every metric returns **NaN rather than a wrong number** on a single-class test
block; `safe_auc` and `safe_mcc` guard that explicitly.

## Regression floors

Two trivial rules, both fitted on the training split only. The floor is the
**lower-MAE** of them, because a model has to beat whichever trivial rule happens
to win on that fold, not the one that flatters it.

| floor | rule |
|---|---|
| constant | the training median wait. Zero rank correlation by construction, so it is the bar that says whether the model learnt anything beyond the base rate. |
| persistence | days-to-next conditioned on days-since-previous, as a **conditional median over deciles**. Usually much the harder of the two. |

Time since the last qualifying event is free, known at prediction time, and the
most informative scalar in the catalogue: after a mainshock the next event is
close (Omori), during a quiet stretch it is far. A model that cannot beat it has
learnt nothing the clock did not already say.

Two choices in that baseline are load-bearing:

- **A conditional median over deciles, not a fitted line.** The relationship is
  monotone but nowhere near linear, and a least-squares fit would understate the
  floor by mismodelling it — making the model look better for a reason that has
  nothing to do with the model.
- **The median, not the mean.** The comparison is MAE, and the median is what
  minimises it. Scoring a mean-optimal baseline on an absolute-error metric is how
  a floor gets quietly lowered.

Hours with no previous event — the opening of the archive — fall back to the
unconditional train median, which is the only thing known about them.

### Spearman is the one to read against the classifier

MAE depends on the censoring cap and on how the wait times happen to fall in a
fold, so it is not comparable across folds, let alone across labels. **The rank
correlation is**, and it answers the same question ROC-AUC answers for the binary
label: does the model order the hours correctly?

A model with a fine MAE and a Spearman near zero has learnt the base rate and
nothing else — which is exactly what predicting the training median achieves.
`regression_report` therefore also reports `pred_std`: a collapsed prediction is
the failure mode of a censored target, and a near-zero spread says so immediately
where MAE alone would look respectable.

`r2` is NaN, not 0, on a constant target: R² is undefined there, and reporting 0
would read as "no better than the mean" when the mean is the only possible answer.

## The warnings the summary raises

`summarise` / `summarise_regression` print these when they apply. Each one marks a
condition under which the headline number means the opposite of what it looks
like.

| warning | condition | why it matters |
|---|---|---|
| `fold spread exceeds the margin over the floor` | `std(auc) > \|mean(auc) − mean(floor)\|` | the pooled number is noise; read the per-fold column |
| `persistence orders the hours better in k/N fold(s)` | model Spearman < persistence Spearman | a model can clear the MAE floor (which is `min(constant, persistence)`) while ordering worse than a free rule. Beating an error bar by being closer to the median is not a forecast. |
| `rank correlation is ~0 across folds` | `mean\|rho\| < 0.05` | the model is not ordering the hours, whatever its MAE says |
| `the prediction is nearly constant` | `pred_std < 0.05 × std(y)` | the model collapsed to the base rate, which an MAE near the constant floor would otherwise hide |
| `most of the target is the cap` | censored fraction > 50% | MAE is mostly measuring the cap |
| `test pos rate is Nx train's` | ratio outside [1/1.5, 1.5] | a swarm or quiet period in one split |

Two NaN-handling details in those checks, both deliberate:

- A NaN model Spearman means the prediction was **constant** — the very failure the
  check warns about — and `nan < 0.43` is `False`, so comparing raw would let it
  through silently. An absent rank correlation is read as ordering nothing, which
  is what it is.
- Every fold's rho can be NaN at once. `nanmean` of all-NaN is NaN with a warning,
  so the ~0-correlation check would skip exactly the run that most needs it; it
  uses the zero-filled array instead.

## Ensembling and seeds

`--ensemble-seeds 42,43,44` by default. The reported score is the mean of the
seeds' probabilities (classify) or predictions (regress), and the **per-seed
spread is printed beside it**. Per-seed AUC spread reaches 0.17 on this data, so a
single-seed number is not a measurement.

`seeding.seed_everything` seeds Python's `random`, NumPy and PyTorch together —
leaving one unseeded would make part of the reported spread an artefact of
ordering rather than of initialisation.
