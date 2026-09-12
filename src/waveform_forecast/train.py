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

    waveform-forecast train --mode regress ...

**No horizon is needed for it, and none should be used lightly.** The target
looks a different distance ahead for every hour -- two days inside an aftershock
sequence, 183 at the longest gap in this catalogue -- so instead of embargoing
the worst case, each hour is purged individually against the block its own next
event falls in (Lopez de Prado Ch. 7, purging by label span). On the MANT+DEMI
split that costs 0.9% of train where a 14-day embargo cost 336 hours.

`--cap-days` censors the wait if a bounded question is wanted, but read the
distribution first: at M>=4.5 the median wait here is 15 days, so a 14-day cap
makes 52% of the target the cap itself and the model is mostly asked to predict
a constant.

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
                                       days_to_next_major,
                                       label_hours_rate_change,
                                       label_resolution_index, label_hours,
                                       load_aegean_events)
from waveform_forecast.data import MultiStationWindows, fit_stats, stack
from waveform_forecast.evaluate import (fold_result, regression_fold_result,
                                        summarise, summarise_regression)
from waveform_forecast.metrics import safe_auc, safe_spearman
from waveform_forecast.model import MultiStationForecaster
from waveform_forecast.regions import (describe_zone, in_window,
                                       load_events_near, load_station_coords,
                                       resolve_stations)
from waveform_forecast.seeding import seed_everything
from waveform_forecast.splits import (print_split_diagnostics,
                                      purge_by_label_span, walk_forward_splits)

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
    p.add_argument("--mode", default="classify",
                   choices=["classify", "regress", "rate"],
                   help="classify: does an M>=threshold event occur within the "
                        "horizon (catalog_mlp's label, unchanged). regress: how "
                        "many DAYS until the next such event, censored at the "
                        "horizon. Same events, same folds, same purge -- the two "
                        "are directly comparable because the censoring makes the "
                        "regression label depend on exactly the same interval. "
                        "rate: will the next window hold MORE events than the "
                        "trailing one -- driven by ~10^3 events at M>=3.5 rather "
                        "than ~10^1 at M>=4.5, which is the only one of the "
                        "three with the sample size for a spatial holdout.")

    g = p.add_argument_group(
        "geography",
        "Where the label comes from, and which stations are held out. The "
        "stations on disk are two tectonic settings, not one: MANT+DEMI are "
        "Aegean (63 km apart) and ELBA+SEMS are Marmara/NAF (114 km apart), "
        "221-295 km away -- and ELBA and SEMS are north of 40N, i.e. OUTSIDE "
        "catalog.AEGEAN_BBOX. Under the region-wide label they are forecasting "
        "a province they cannot see.")
    g.add_argument("--label-radius-km", type=float, default=None,
                   help="label each split's hours with events within this many "
                        "km of the NEAREST station forecasting them, instead of "
                        "the region-wide AEGEAN_BBOX. Required for a spatial "
                        "holdout to mean anything: without it both zones carry "
                        "the same region-wide label and the 'held-out' zone is "
                        "predicting the training zone's earthquakes.")
    g.add_argument("--test-stations", nargs="+", default=None,
                   help="hold these stations out as the test zone, e.g. "
                        "`--stations aegean --test-stations marmara`. Defaults "
                        "to --stations, i.e. no spatial holdout. Zone names "
                        "(aegean, marmara) expand to their station codes.")
    g.add_argument("--station-table", default=None,
                   help="AFAD istasyon_katalog.csv for station coordinates. "
                        "Omit to use the built-in coordinates for MANT, DEMI, "
                        "ELBA and SEMS.")

    g = p.add_argument_group("rate mode")
    g.add_argument("--rate-threshold", type=float, default=3.5,
                   help="magnitude defining the events whose RATE is forecast. "
                        "Much lower than --threshold on purpose: the point of "
                        "rate mode is a label driven by many events.")
    g.add_argument("--baseline-days", type=float, default=None,
                   help="trailing window the forward window is compared "
                        "against. Defaults to --horizon-days, a like-for-like "
                        "comparison with no window-length bias.")
    p.add_argument("--cap-days", type=float, default=None,
                   help="regress only: censor the target at this many days, so "
                        "an hour whose next event is further off is recorded as "
                        '"at least this long". OPTIONAL -- the split is kept '
                        "honest by purging each hour against the block its own "
                        "next event falls in, not by a fixed horizon, so an "
                        "uncapped target is fine. Cap only to ask a bounded "
                        "question: at M>=4.5 the median wait is 15 d, so a 14 d "
                        "cap makes 52% of the target the cap itself.")
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

    zones = resolve_zones(args)
    wanted = [c for c, _ in zones["all"]]
    missing = [s for s in wanted if f"present_{s}" not in frame.columns]
    if missing:
        have = sorted(c[len("present_"):] for c in frame.columns
                      if c.startswith("present_"))
        sys.exit(f"[ERROR] {missing} not in this table; it holds {have}")

    if args.keep_features is not None:
        keep = set(args.keep_features)
        drop = [c for c in frame.columns
                if "__" in c and c.split("__", 1)[1] not in keep]
        frame = frame.drop(columns=drop)

    x, present, feats = stack(frame, wanted)
    if args.keep_features is not None:
        unknown = [f for f in args.keep_features if f not in feats]
        if unknown:
            sys.exit(f"[ERROR] --keep-features got {unknown}, which this table "
                     f"does not have. It holds {len(feats)} columns, e.g. "
                     f"{feats[:4]}")

    labels = {name: zone_labels(args, hour_index, zones[name], name)
              for name in ("train", "test")}
    return x, present, labels, hour_index, feats, zones


def resolve_zones(args):
    """Which stations train, which stations are tested, and where they are.

    Returns:
        Dict with "train"/"test" -> [(code, (lat, lon)), ...], "all" -> their
        union in a fixed order, and "idx" -> {name: positions into "all"}. The
        union is what gets stacked, so both zones sit on one hour grid and one
        set of standardization statistics.
    """
    coords = load_station_coords(args.station_table)
    train = resolve_stations(args.stations, coords)
    test = resolve_stations(args.test_stations or args.stations, coords)
    order, seen = [], set()
    for code, xy in train + test:
        if code not in seen:
            seen.add(code)
            order.append((code, xy))
    pos = {c: i for i, (c, _) in enumerate(order)}
    return {"train": train, "test": test, "all": order,
            "idx": {"train": np.array([pos[c] for c, _ in train]),
                    "test": np.array([pos[c] for c, _ in test])}}


def zone_labels(args, hour_index, zone, name):
    """The label, its floor inputs, and its purge index, for one zone.

    **The whole point of the region-local path.** Each zone is labelled by the
    events its OWN stations could plausibly have recorded. With
    `--label-radius-km` unset this collapses to the region-wide AEGEAN_BBOX
    label and both zones get the same one, which is the pre-existing behaviour
    and the right thing when every station sits inside the box.

    Returns:
        Dict with "labels", "dsp", "censored", "resolves", "trailing", "times".
    """
    codes = [c for c, _ in zone]
    centers = [xy for _, xy in zone]
    if args.label_radius_km:
        major = load_events_near(args.catalog_path, args.threshold, centers,
                                 args.label_radius_km)
    else:
        major = load_aegean_events(args.catalog_path, args.threshold)
    describe_zone(f"{name} zone", codes, centers, major,
                  args.label_radius_km, hour_index)
    if not len(major):
        sys.exit(f"[ERROR] no M>={args.threshold} event to label the {name} "
                 f"zone with. Widen --label-radius-km, lower --threshold, or "
                 f"drop --label-radius-km for the region-wide label.")

    dsp = days_since_prev_major(hour_index, major)
    out = {"labels": None, "dsp": dsp, "censored": None, "resolves": None,
           "trailing": None, "times": major}

    if args.mode == "rate":
        # A separate, much lower threshold: the label is a COUNT comparison, so
        # it wants the completeness-limited catalogue rather than the handful of
        # qualifying events. Same zone, same radius.
        if args.label_radius_km:
            rate_times = load_events_near(args.catalog_path,
                                          args.rate_threshold, centers,
                                          args.label_radius_km)
        else:
            rate_times = load_aegean_events(args.catalog_path,
                                            args.rate_threshold)
        if not len(rate_times):
            sys.exit(f"[ERROR] no M>={args.rate_threshold} event in the {name} "
                     f"zone; there is no rate to forecast")
        lab, fwd, bwd = label_hours_rate_change(hour_index, rate_times,
                                                args.horizon_days,
                                                args.baseline_days)
        base = args.baseline_days or args.horizon_days
        print(f"      rate label from {len(in_window(rate_times, hour_index)):,} "
              f"M>={args.rate_threshold:g} event(s) in span; forward "
              f"{args.horizon_days:g} d vs trailing {base:g} d")
        print(f"      positive rate {lab.mean():.3f} "
              f"(1 = the next window holds more events than the last)")
        out["labels"], out["trailing"] = lab, bwd
        return out

    if args.mode == "regress":
        lab, censored = days_to_next_major(hour_index, major, args.cap_days)
        n_open = int(np.isnan(lab).sum())
        if n_open:
            print(f"      {n_open:,} hour(s) past the last M>={args.threshold} "
                  f"event ({major[-1]}) have no next event and are dropped —\n"
                  f"          right-censored, not quiet")
        if args.cap_days is not None:
            n_lab = max(int(np.isfinite(lab).sum()), 1)
            print(f"      {censored.sum():,} of {n_lab:,} labelled hour(s) wait "
                  f"longer than the {args.cap_days:g} d cap "
                  f"({100 * censored.sum() / n_lab:.1f}%)")
        out["labels"], out["censored"] = lab, censored
        # When each hour's label settles. This replaces the horizon term in the
        # embargo: the lookahead is per hour, so the purge is too.
        out["resolves"] = label_resolution_index(hour_index, major,
                                                 args.cap_days)
        return out

    out["labels"] = label_hours(hour_index, major, args.horizon_days)
    return out


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


def run_fold(label, args, x, present, lab, hour_index, tr_i, va_i, te_i,
             seeds, device, feat_dim, zones, header=True):
    """Trains the ensemble on one split and scores it against that fold's floor.

    Args:
        lab: {"train": zone bundle, "test": zone bundle} from `zone_labels`.
            Train and val read the TRAIN zone's label, test reads the TEST
            zone's. Without a spatial holdout the two bundles are identical and
            this is the previous behaviour exactly.
        zones: From `resolve_zones` -- supplies the station sub-axis each split
            uses.
    """
    regress = args.mode == "regress"
    tr_lab, te_lab = lab["train"], lab["test"]
    spatial = args.test_stations is not None
    if header:
        print(f"\n{'=' * 64}\n{label}\n{'=' * 64}")
    print(f"  splits (chronological): train={len(tr_i)} val={len(va_i)} test={len(te_i)}")
    for name, idx, src in (("train", tr_i, tr_lab), ("val", va_i, tr_lab),
                           ("test", te_i, te_lab)):
        if not len(idx):
            continue
        if regress:
            print(f"    {name:5s}: median wait {np.median(src['labels'][idx]):6.2f} d   "
                  f"censored {100 * src['censored'][idx].mean():5.1f}%")
        else:
            print(f"    {name:5s}: positive rate {src['labels'][idx].mean():.3f}")
    print_split_diagnostics(hour_index, tr_lab["labels"], tr_i, va_i, te_i,
                            quantity="mean wait (d)" if regress else "pos rate")
    # Which stations the fold actually had. A "multi-station" result whose test
    # block was one station is a single-station result under another name -- and
    # with a held-out zone, a test block where the held-out stations are down is
    # not a spatial-transfer measurement at all.
    codes = [c for c, _ in zones["test"]]
    cov = present[np.ix_(te_i, zones["idx"]["test"])].mean(axis=0)
    for s, c in zip(codes, cov):
        print(f"    test coverage {s:6s}: {c * 100:5.1f}% of hours")
    if spatial and cov.max() < 0.05:
        print("  [!] the held-out zone is absent from this test block -- there "
              "is nothing to\n      transfer TO here. Skipping the fold.")
        return None

    if len(tr_i) < 10 or len(te_i) < 5:
        print("[ERROR] not enough data for a meaningful split")
        return None

    # Every station's statistics, fitted over the training TIME window. See
    # `fit_stats`: with a held-out zone the test stations appear in no training
    # window, and identity-standardizing them would measure a scaling mismatch
    # rather than transfer.
    stats = fit_stats(x, present, tr_i, args.seq_hours)
    # The dataset carries the FITTED space; `train_one_seed` inverts before it
    # reports. Transforming here rather than inside the dataset keeps the
    # inversion and the transform one function apart from each other, so a
    # change to one is visibly a change to the other.
    def fitted(src):
        return (transform_target(src["labels"], args.target_transform)
                if regress else src["labels"])

    def mk(idx, src, which):
        return MultiStationWindows(x, present, fitted(src), args.seq_hours, idx,
                                   stats, station_idx=zones["idx"][which])

    per_seed, yt_ref = [], None
    for seed in seeds:
        yt, st = train_one_seed(args, seed,
                                mk(tr_i, tr_lab, "train"),
                                mk(va_i, tr_lab, "train"),
                                mk(te_i, te_lab, "test"),
                                feat_dim, device)
        if yt_ref is None:
            yt_ref = yt
        per_seed.append(st)

    name = f"multistation[{args.arm}]"
    if spatial:
        name += f" {'+'.join(c for c, _ in zones['train'])}->{'+'.join(codes)}"
    if regress:
        return regression_fold_result(
            label, yt_ref, np.mean(per_seed, axis=0), per_seed,
            y_train=tr_lab["labels"][tr_i], dsp_train=tr_lab["dsp"][tr_i],
            dsp_test=te_lab["dsp"][te_i], censored=te_lab["censored"][te_i],
            model_name=name)
    if args.mode == "rate":
        # The trailing count is the floor, and it must come from the zone being
        # SCORED -- the test zone's own trailing seismicity is what a free rule
        # would have had. Taking it from the train zone would score the model
        # against a baseline built on data it is being tested away from.
        return fold_result(label, yt_ref, np.mean(per_seed, axis=0), per_seed,
                           tr_lab["labels"][tr_i], "rate", te_lab["dsp"][te_i],
                           args.horizon_days,
                           rate_trailing_test=te_lab["trailing"][te_i],
                           rate_trailing_train=tr_lab["trailing"][tr_i],
                           model_name=name)
    return fold_result(label, yt_ref, np.mean(per_seed, axis=0), per_seed,
                       tr_lab["labels"][tr_i], "event", te_lab["dsp"][te_i],
                       args.horizon_days, model_name=name)


def run(args):
    x, present, lab, hour_index, feats, zones = build_inputs(args)
    n, n_st, n_feat = x.shape
    tr_lab, te_lab = lab["train"], lab["test"]
    spatial = args.test_stations is not None
    print(f"\n  {n:,} hourly rows, {n_st} station(s), {n_feat} feature(s) each")
    if args.mode == "regress":
        ok = np.isfinite(tr_lab["labels"])
        cap = (f"censored at {args.cap_days:g} d" if args.cap_days is not None
               else "uncapped; the split is purged per hour instead")
        print(f"  wait to next M>={args.threshold}: median "
              f"{np.median(tr_lab['labels'][ok]):.2f} d, mean "
              f"{np.mean(tr_lab['labels'][ok]):.2f} d ({cap})")
    else:
        print(f"  hourly positive rate: train zone "
              f"{tr_lab['labels'].mean():.3f}"
              + (f", test zone {te_lab['labels'].mean():.3f}" if spatial else ""))
    for k, (s, _) in enumerate(zones["all"]):
        print(f"    {s:6s} present {present[:, k].mean() * 100:5.1f}% of hours")

    # Presence is reported for the STATION SET each split actually uses. With a
    # held-out zone the union's "any present" would be dominated by stations the
    # test split never sees, which is the number that would make a spatial run
    # look better covered than it is.
    for which in (("train", "test") if spatial else ("train",)):
        k = zones["idx"][which]
        pres = present[:, k]
        tag = f"{which} zone" if spatial else "all stations"
        print(f"    {tag}: any {pres.any(1).mean() * 100:5.1f}%   "
              f"all {pres.all(1).mean() * 100:5.1f}%")

    # A window is usable when its LAST hour -- the one the label attaches to --
    # has enough stations. An earlier hour being empty is what the mask is for.
    # Each split is judged on ITS OWN zone: a window with only training stations
    # up is not a usable test window in a spatial run.
    def usable(which, src):
        k = zones["idx"][which]
        enough = present[:, k].sum(axis=1) >= args.min_stations
        v = np.arange(args.seq_hours - 1, n)
        v = v[enough[v]]
        if args.mode == "regress":
            # The tail of the archive has no next event, so its target is
            # unknown rather than long. Dropping those hours is what keeps the
            # model from learning that the record ends -- an imputed large value
            # would make the quietest-looking stretch the one just before it
            # stops.
            before = len(v)
            v = v[np.isfinite(src["labels"][v])]
            if before != len(v):
                print(f"  {which}: dropped {before - len(v):,} right-censored "
                      f"window(s) past the last qualifying event")
        return v

    valid_tr = usable("train", tr_lab)
    valid_te = usable("test", te_lab) if spatial else valid_tr
    if not len(valid_tr) or not len(valid_te):
        sys.exit(f"[ERROR] no window has >= {args.min_stations} station(s) at "
                 f"its last hour" + (" with a known wait to the next event"
                                     if args.mode == "regress" else ""))
    print(f"  {len(valid_tr):,} usable train window(s)"
          + (f", {len(valid_te):,} usable test window(s)" if spatial else "")
          + f" of {n - args.seq_hours + 1:,}")

    # seq_hours-1 removes INPUT overlap at a block boundary: a window ending
    # just inside val reads hours that belong to train.
    #
    # The LABEL's lookahead is handled differently by mode. In classify and rate
    # it is exactly horizon_days for every hour, so a constant term in the
    # embargo is exact (Lopez de Prado, Ch. 7). In regress it is the distance to
    # the next event, which is different for every hour -- 2 days inside an
    # aftershock sequence, 183 at the longest gap in this catalogue. Embargoing
    # the worst case would cost a quarter of a two-year archive to protect the
    # few hours that need it, so those hours are purged individually below.
    embargo = args.seq_hours - 1
    if args.mode != "regress":
        embargo += int(round(args.horizon_days * 24))

    # The fold GEOMETRY comes from the training zone's usable windows, so a
    # spatial run's blocks are the same stretches of time as its in-zone
    # counterpart and the two are comparable. The test block is then whatever
    # the test zone has inside that stretch.
    if args.cv_folds <= 1:
        i_tr = int(len(valid_tr) * args.train_frac)
        i_va = int(len(valid_tr) * (args.train_frac + args.val_frac))
        folds = [(valid_tr[:i_tr], valid_tr[i_tr + embargo:i_va],
                  valid_tr[i_va + embargo:])]
        names = ["single split"]
    else:
        folds = walk_forward_splits(valid_tr, args.cv_folds, embargo=embargo)
        names = [f"fold {k + 1}/{args.cv_folds}" for k in range(args.cv_folds)]

    if spatial:
        # Keep only test indices the TEST zone can actually supply, without
        # moving the boundary -- so the block stays the same span of time and
        # the temporal separation the embargo bought is untouched.
        keep = set(valid_te.tolist())
        folds = [(tr, va, te[np.isin(te, valid_te)]) for tr, va, te in folds]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.ensemble_seeds.split(",")]
    print(f"  device {device}, seeds {seeds}, arm={args.arm}, "
          f"mode={args.mode}, embargo {embargo}h")
    if spatial:
        print(f"  SPATIAL HOLDOUT: train on "
              f"{'+'.join(c for c, _ in zones['train'])}, test on "
              f"{'+'.join(c for c, _ in zones['test'])}"
              + ("" if args.label_radius_km else
                 "\n  [!] --label-radius-km is unset, so BOTH zones carry the "
                 "same region-wide label.\n      The held-out zone is being "
                 "scored on the training zone's earthquakes."))

    results = []
    for name, (tr_i, va_i, te_i) in zip(names, folds):
        if args.mode == "regress":
            print(f"\n{'=' * 64}\n{name}\n{'=' * 64}")
            # Train/val purge against the TRAIN zone's label spans, test against
            # its own -- the two zones' events settle at different times.
            tr_i, va_i, _ = purge_by_label_span(tr_i, va_i, te_i,
                                                tr_lab["resolves"])
            if not len(tr_i) or not len(te_i):
                print("  nothing survives the purge on this fold")
                continue
        r = run_fold(name, args, x, present, lab, hour_index,
                     tr_i, va_i, te_i, seeds, device, n_feat, zones,
                     header=args.mode != "regress")
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
