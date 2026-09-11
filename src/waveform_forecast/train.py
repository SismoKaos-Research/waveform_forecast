"""Train the multi-station forecaster, and score it against its floor.

    waveform-forecast train \
        --features hourly_mant_demi.parquet --stations MANT DEMI \
        --catalog-path ../cnn_earthquake/catalogs/catalog_current.csv \
        --horizon-days 14 --cv-folds 5

**The label is `../forecast`'s, unchanged.** `label_hours` over the same
catalogue, the same `walk_forward_splits`, the same purge, and the same
`evaluate.fold_result`. That is what makes a waveform result and a catalogue
result comparable rather than merely adjacent: catalog_mlp answers "does an
M>=threshold event occur within the horizon" from catalogue features, and this
answers the same question from several stations' waveforms.

**Two arms, one argument apart.** `--arm features` pools the hourly feature
vectors directly. `--arm raw` puts a 1D CNN in front of each (station, hour) of
5 Hz samples first. Everything else -- splits, purge, floor, folds -- is shared,
which is the only way the two are a comparison.

**Two questions, also one argument apart.** `--mode classify` is the label
above. `--mode regress` asks how many DAYS until the next M>=threshold event,
censored at the same horizon:

    waveform-forecast train --mode regress --horizon-days 14 ...

The censoring is not a convenience. An uncapped "days to next" depends on an
event arbitrarily far ahead, so a training hour near a fold boundary carries a
label decided inside the validation block and no finite embargo removes it.
Capping at the horizon bounds the lookahead to exactly the interval the binary
label already uses -- which is also what makes the two modes comparable on the
same folds rather than merely adjacent.

Its floor is not 0.5 but two trivial rules in days: the training median, and
days-to-next conditioned on days-since-previous. The second is usually much the
harder. Time since the last event is free, known at prediction time, and the
most informative scalar in the catalogue; a model that cannot beat it has learnt
nothing the clock did not already say.

**The floor is the point.** The waveform arms are a documented negative in this
project: on the corrected catalogue neither raw-waveform CNNs nor hand-crafted
continuous features beat persistence, in 0 of 10 chaos sweep cells and across
all three sequence architectures. Multi-station is a real difference from those
runs, and it may or may not matter -- but it is measured against the same bar,
per fold, with the spread beside it.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from waveform_forecast.catalog import (days_since_prev_major,
                                       days_to_next_major, label_hours,
                                       load_aegean_events)
from waveform_forecast.data import MultiStationWindows, fit_stats, stack
from waveform_forecast.evaluate import (fold_result, regression_fold_result,
                                        summarise, summarise_regression)
from waveform_forecast.metrics import safe_auc, safe_spearman
from waveform_forecast.model import MultiStationForecaster
from waveform_forecast.seeding import seed_everything
from waveform_forecast.splits import print_split_diagnostics, walk_forward_splits

NAME = "train"
HELP = "train the multi-station forecaster and score it against its floor"


def add_args(p):
    p.add_argument("--features", required=True,
                   help="the parquet `waveform-forecast features` wrote")
    p.add_argument("--stations", nargs="+", required=True,
                   help="station codes to use, as named in that table")
    p.add_argument("--catalog-path", required=True,
                   help="catalogue CSV -- the LABEL, not an input")
    p.add_argument("--arm", default="features", choices=["features", "raw"],
                   help="features: pool the hourly vectors. raw: a 1D CNN over "
                        "one hour of 5 Hz samples per station first.")
    p.add_argument("--mode", default="classify", choices=["classify", "regress"],
                   help="classify: does an M>=threshold event occur within the "
                        "horizon (catalog_mlp's label, unchanged). regress: how "
                        "many DAYS until the next such event, censored at the "
                        "horizon. Same events, same folds, same purge -- the two "
                        "are directly comparable because the censoring makes the "
                        "regression label depend on exactly the same interval.")
    p.add_argument("--target-transform", default="log1p", choices=["log1p", "none"],
                   help="log1p: fit on log1p(days) and invert for reporting. The "
                        "wait distribution is heavy-tailed, and a plain squared "
                        "error on raw days is dominated by the longest waits -- "
                        "the model spends its capacity on the tail and predicts "
                        "the mean everywhere else. Metrics are always reported "
                        "in DAYS either way.")
    p.add_argument("--threshold", type=float, default=4.5)
    p.add_argument("--horizon-days", type=float, default=14.0)
    p.add_argument("--seq-hours", type=int, default=24)
    p.add_argument("--keep-features", nargs="+", default=None,
                   help="restrict to these feature columns by name, e.g. "
                        "Z_STA_LTA_Max_max. The table is ~200 columns per "
                        "station and the positive class is a handful of events "
                        "per fold, so this is usually not optional.")
    p.add_argument("--min-stations", type=int, default=1,
                   help="drop windows whose LAST hour has fewer than this many "
                        "stations present")

    g = p.add_argument_group("model")
    g.add_argument("--hidden", type=int, default=64)
    g.add_argument("--proj-dim", type=int, default=None)
    g.add_argument("--dropout", type=float, default=0.3)

    g = p.add_argument_group("training")
    g.add_argument("--epochs", type=int, default=40)
    g.add_argument("--batch-size", type=int, default=64)
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--weight-decay", type=float, default=0.1)
    g.add_argument("--patience", type=int, default=8)
    g.add_argument("--ensemble-seeds", default="42,43,44")
    g.add_argument("--num-workers", type=int, default=2)

    g = p.add_argument_group("evaluation")
    g.add_argument("--cv-folds", type=int, default=5)
    g.add_argument("--train-frac", type=float, default=0.70)
    g.add_argument("--val-frac", type=float, default=0.15)
    return p


def build_inputs(args):
    """Reads the hourly table, joins the label, and returns the model's inputs.

    Returns:
        (x, present, labels, dsp, hour_index, feature_names).
    """
    frame = pd.read_parquet(args.features).sort_index()
    if not isinstance(frame.index, pd.DatetimeIndex):
        sys.exit(f"[ERROR] {args.features} is not indexed by hour; was it "
                 f"written by `waveform-forecast features`?")
    hour_index = pd.DatetimeIndex(frame.index)
    # The features table carries an explicit UTC index, which is right for an
    # artifact. The catalogue is parsed tz-naive ("%d/%m/%Y %H:%M:%S") and every
    # label function in ../forecast works in naive UTC, so the two must be put
    # on one convention before they meet. Converting to UTC first and only then
    # dropping the marker is the lossless direction: `tz_localize(None)` alone
    # on a non-UTC index would silently shift every label by the offset.
    if hour_index.tz is not None:
        hour_index = hour_index.tz_convert("UTC").tz_localize(None)

    missing = [s for s in args.stations if f"present_{s}" not in frame.columns]
    if missing:
        have = sorted(c[len("present_"):] for c in frame.columns
                      if c.startswith("present_"))
        sys.exit(f"[ERROR] {missing} not in this table; it holds {have}")

    if args.keep_features is not None:
        keep = set(args.keep_features)
        drop = [c for c in frame.columns
                if "__" in c and c.split("__", 1)[1] not in keep]
        frame = frame.drop(columns=drop)

    x, present, feats = stack(frame, args.stations)
    if args.keep_features is not None:
        unknown = [f for f in args.keep_features if f not in feats]
        if unknown:
            sys.exit(f"[ERROR] --keep-features got {unknown}, which this table "
                     f"does not have. It holds {len(feats)} columns, e.g. "
                     f"{feats[:4]}")

    # The catalogue supplies the answer key only. Restricting it to the
    # stations' own neighbourhood is deliberately NOT done: the label is
    # region-wide, exactly as it is in catalog_mlp, or the two are not the
    # same question.
    major = load_aegean_events(args.catalog_path, args.threshold)
    if not len(major):
        sys.exit(f"[ERROR] no M>={args.threshold} event in {args.catalog_path} "
                 f"inside the Aegean box; there is nothing to label with")
    dsp = days_since_prev_major(hour_index, major)
    if args.mode == "regress":
        # Censored at the horizon so the label's lookahead is bounded and the
        # walk-forward purge can remove it. See `days_to_next_major`.
        labels, censored = days_to_next_major(hour_index, major, args.horizon_days)
        n_open = int(np.isnan(labels).sum())
        if n_open:
            print(f"  {n_open:,} hour(s) past the last M>={args.threshold} event "
                  f"({major[-1]}) have no next event and are dropped —\n"
                  f"      right-censored, not quiet")
        print(f"  {censored.sum():,} of {np.isfinite(labels).sum():,} labelled "
              f"hour(s) wait longer than the {args.horizon_days:g} d cap "
              f"({100 * censored.sum() / max(np.isfinite(labels).sum(), 1):.1f}%)")
        return x, present, labels, dsp, hour_index, feats, censored
    labels = label_hours(hour_index, major, args.horizon_days)
    return x, present, labels, dsp, hour_index, feats, None


def train_one_seed(args, seed, ds_train, ds_val, ds_test, feat_dim, device):
    """Trains one seed and returns (y_true, y_score) on the test split."""
    seed_everything(seed)
    encoder = None
    if args.arm == "raw":
        from waveform_forecast.blocks import RawWaveformEncoder
        encoder = RawWaveformEncoder(out_dim=args.proj_dim or 32,
                                     dropout=args.dropout)
        feat_dim = encoder.out_dim
    model = MultiStationForecaster(feat_dim=feat_dim, hidden=args.hidden,
                                   dropout=args.dropout, encoder=encoder,
                                   proj_dim=args.proj_dim).to(device)

    def dl(ds, sh):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                          num_workers=args.num_workers)

    tr, va, te = dl(ds_train, True), dl(ds_val, False), dl(ds_test, False)
    if args.mode == "regress":
        # Huber, not MSE. Even on log1p days the tail is long, and a squared
        # error lets the handful of longest waits set the gradient for every
        # batch they land in -- the classic outcome is a model that predicts the
        # mean everywhere and reports a respectable MAE with zero rank
        # correlation, which `regression_report`'s `pred_std` is there to catch.
        criterion = nn.HuberLoss(delta=1.0)
    else:
        pos = float(np.mean([ds_train.labels[i] for i in ds_train.indices]))
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor((1 - pos) / max(pos, 1e-6), dtype=torch.float32,
                                    device=device))
    opt = optim.AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    regress = args.mode == "regress"

    def score(loader):
        """Returns (y_true, y_pred, mean_loss) for one split.

        The loss comes from the SAME criterion the optimiser saw, pos_weight
        included. A val loss computed with an unweighted BCE would be on a
        different scale from the training objective and could not be read
        against it, which is most of what a val loss is for.

        In regression mode the loss stays in the TRANSFORMED space the model
        was fitted in, while `y_true` and `y_pred` come back in days. Mixing
        those is how a log-space MAE of 0.4 gets read as "within half a day".
        """
        model.eval()
        ys, ss, total, seen = [], [], 0.0, 0
        with torch.no_grad():
            for w, m, y in loader:
                yd = y.to(device)
                out = model(w.to(device), m.to(device))
                # Weighted by batch size, not averaged over batches: the last
                # batch is usually short, and a plain mean of batch means lets
                # its handful of windows count as much as a full batch.
                total += float(criterion(out, yd)) * len(y)
                seen += len(y)
                ss.extend((out if regress else torch.sigmoid(out)).cpu().tolist())
                ys.extend(y.tolist())
        y = np.array(ys, dtype=np.float64 if regress else np.int64)
        pred = np.array(ss, dtype=np.float64)
        if regress:
            y, pred = invert(y), invert(pred)
        return y, pred, total / seen if seen else float("nan")

    def invert(v):
        """Back to days from whatever space the model was fitted in."""
        if args.target_transform != "log1p":
            return v
        # clip before expm1: an unconstrained head can emit a large negative
        # early in training, and expm1 of that is -1 day, which then poisons
        # the MAE with a physically impossible wait.
        return np.expm1(np.clip(v, 0.0, 50.0))

    # -inf, not -1: in regression the tracked metric is a NEGATIVE MAE, and
    # -1.0 would silently reject every checkpoint worse than one day.
    best, no_improve, best_state = float("-inf"), 0, None
    for epoch in range(args.epochs):
        model.train()
        run_loss, run_n = 0.0, 0
        for w, m, y in tr:
            loss = criterion(model(w.to(device), m.to(device)), y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            # detach, or reading the scalar keeps the graph alive for the
            # whole epoch's worth of batches -- a slow memory leak on the
            # raw arm, where one window is 3 x 18,000 samples.
            run_loss += float(loss.detach()) * len(y)
            run_n += len(y)
        sched.step()
        yv, sv, val_loss = score(va)
        # Both losses, because a val loss alone cannot separate a model that is
        # learning slowly from one that has started memorising -- and the
        # headline metric alone hides both. The positive class here is a handful
        # of events per fold, so `safe_auc` returns NaN whenever a val block
        # happens to hold one class; the loss stays finite and readable when
        # that happens, which is exactly the epoch where a number is wanted.
        head = f"  [seed {seed}] epoch {epoch + 1}/{args.epochs}" \
               f"  train loss {run_loss / max(run_n, 1):.4f}" \
               f"  val loss {val_loss:.4f}"
        if regress:
            # Selection on MAE in DAYS, not on the transformed loss: the loss
            # is what the optimiser minimises, but the checkpoint should be the
            # one that is best at the thing being reported. Spearman rides
            # along because a flat prediction can hold a fine MAE.
            val_mae = float(np.mean(np.abs(sv - yv)))
            metric, better = -val_mae, (-val_mae > best)
            print(f"{head}  val MAE {val_mae:.3f} d"
                  f"  val rho {safe_spearman(yv, sv):+.4f}")
        else:
            val_auc = safe_auc(yv, sv)
            metric, better = val_auc, (np.isfinite(val_auc) and val_auc > best)
            print(f"{head}  val AUC {val_auc:.4f}")
        if better:
            best, no_improve = metric, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    yt, st, test_loss = score(te)
    if regress:
        print(f"  [seed {seed}] test loss {test_loss:.4f}  "
              f"test MAE {np.mean(np.abs(st - yt)):.3f} d  "
              f"test rho {safe_spearman(yt, st):+.4f}")
    else:
        print(f"  [seed {seed}] test loss {test_loss:.4f}  "
              f"test AUC {safe_auc(yt, st):.4f}")
    return yt, st


def transform_target(labels, how):
    """Days -> the space the model is fitted in. `invert` undoes it."""
    if how != "log1p":
        return labels
    # log1p, not log: the wait is zero at the instant of an event and log(0) is
    # -inf. log1p is also near-linear under a day, where the differences that
    # matter to a forecast actually live.
    return np.log1p(np.clip(labels, 0.0, None))


def run_fold(label, args, x, present, labels, dsp, hour_index, tr_i, va_i, te_i,
             seeds, device, feat_dim, censored=None):
    """Trains the ensemble on one split and scores it against that fold's floor."""
    regress = args.mode == "regress"
    print(f"\n{'=' * 64}\n{label}\n{'=' * 64}")
    print(f"  splits (chronological): train={len(tr_i)} val={len(va_i)} test={len(te_i)}")
    for name, idx in (("train", tr_i), ("val", va_i), ("test", te_i)):
        if not len(idx):
            continue
        if regress:
            print(f"    {name:5s}: median wait {np.median(labels[idx]):6.2f} d   "
                  f"censored {100 * censored[idx].mean():5.1f}%")
        else:
            print(f"    {name:5s}: positive rate {labels[idx].mean():.3f}")
    print_split_diagnostics(hour_index, labels, tr_i, va_i, te_i,
                            quantity="mean wait (d)" if regress else "pos rate")
    # Which stations the fold actually had. A "multi-station" result whose test
    # block was one station is a single-station result under another name.
    for k, s in enumerate(args.stations):
        cov = present[te_i, k].mean()
        print(f"    test coverage {s:6s}: {cov * 100:5.1f}% of hours")

    if len(tr_i) < 10 or len(te_i) < 5:
        print("[ERROR] not enough data for a meaningful split")
        return None

    stats = fit_stats(x, present, tr_i, args.seq_hours)
    # The dataset carries the FITTED space; `train_one_seed` inverts before it
    # reports. Transforming here rather than inside the dataset keeps the
    # inversion and the transform one function apart from each other, so a
    # change to one is visibly a change to the other.
    fitted = transform_target(labels, args.target_transform) if regress else labels
    mk = lambda idx: MultiStationWindows(x, present, fitted, args.seq_hours,
                                         idx, stats)
    per_seed, yt_ref = [], None
    for seed in seeds:
        yt, st = train_one_seed(args, seed, mk(tr_i), mk(va_i), mk(te_i),
                                feat_dim, device)
        if yt_ref is None:
            yt_ref = yt
        per_seed.append(st)

    name = f"multistation[{args.arm}]"
    if regress:
        return regression_fold_result(
            label, yt_ref, np.mean(per_seed, axis=0), per_seed,
            y_train=labels[tr_i], dsp_train=dsp[tr_i], dsp_test=dsp[te_i],
            censored=censored[te_i], model_name=name)
    return fold_result(label, yt_ref, np.mean(per_seed, axis=0), per_seed,
                       labels[tr_i], "event", dsp[te_i], args.horizon_days,
                       model_name=name)


def run(args):
    x, present, labels, dsp, hour_index, feats, censored = build_inputs(args)
    n, n_st, n_feat = x.shape
    print(f"  {n:,} hourly rows, {n_st} station(s), {n_feat} feature(s) each")
    if args.mode == "regress":
        ok = np.isfinite(labels)
        print(f"  wait to next M>={args.threshold}: median "
              f"{np.median(labels[ok]):.2f} d, mean {np.mean(labels[ok]):.2f} d "
              f"(censored at {args.horizon_days:g} d)")
    else:
        print(f"  hourly positive rate: {labels.mean():.3f}")
    for k, s in enumerate(args.stations):
        print(f"    {s:6s} present {present[:, k].mean() * 100:5.1f}% of hours")
    print(f"    {'any':6s} present {present.any(1).mean() * 100:5.1f}%   "
          f"<- what the masked model trains on")
    print(f"    {'all':6s} present {present.all(1).mean() * 100:5.1f}%   "
          f"<- what an intersection-only model would get")

    # A window is usable when its LAST hour -- the one the label attaches to --
    # has enough stations. An earlier hour being empty is what the mask is for.
    enough = present.sum(axis=1) >= args.min_stations
    valid = np.arange(args.seq_hours - 1, n)
    valid = valid[enough[valid]]
    if args.mode == "regress":
        # The tail of the archive has no next event, so its target is unknown
        # rather than long. Dropping those hours is what keeps the model from
        # learning that the record ends -- an imputed large value would make the
        # quietest-looking stretch in the data the one just before it stops.
        before = len(valid)
        valid = valid[np.isfinite(labels[valid])]
        if before != len(valid):
            print(f"  dropped {before - len(valid):,} right-censored window(s) "
                  f"past the last qualifying event")
    if not len(valid):
        sys.exit(f"[ERROR] no window has >= {args.min_stations} station(s) at "
                 f"its last hour" + (" with a known wait to the next event"
                                     if args.mode == "regress" else ""))
    print(f"  {len(valid):,} usable window(s) of {n - args.seq_hours + 1:,}")

    # seq_hours-1 removes input overlap at a block boundary; the label looks
    # horizon_days forward, so without the horizon term the last ~14 days of
    # every block carry labels decided by events inside the NEXT block -- train
    # labels encoding what happens in val (Lopez de Prado, Ch. 7).
    embargo = args.seq_hours - 1 + int(round(args.horizon_days * 24))

    if args.cv_folds <= 1:
        i_tr = int(len(valid) * args.train_frac)
        i_va = int(len(valid) * (args.train_frac + args.val_frac))
        folds = [(valid[:i_tr], valid[i_tr + embargo:i_va], valid[i_va + embargo:])]
        names = ["single split"]
    else:
        folds = walk_forward_splits(valid, args.cv_folds, embargo=embargo)
        names = [f"fold {k + 1}/{args.cv_folds}" for k in range(args.cv_folds)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.ensemble_seeds.split(",")]
    print(f"  device {device}, seeds {seeds}, arm={args.arm}, "
          f"mode={args.mode}, embargo {embargo}h")

    results = []
    for name, (tr_i, va_i, te_i) in zip(names, folds):
        r = run_fold(name, args, x, present, labels, dsp, hour_index,
                     tr_i, va_i, te_i, seeds, device, n_feat, censored)
        if r is not None:
            results.append(r)
    if args.cv_folds > 1:
        if args.mode == "regress":
            summarise_regression(results, args.cv_folds)
        else:
            summarise(results, args.cv_folds)
    return 0


def main():
    p = argparse.ArgumentParser(prog="waveform-forecast train", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(p)
    return run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
