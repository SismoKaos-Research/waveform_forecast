"""What the epoch line has to say, and the epoch where it has to say it.

Selection is on val AUC and that has not changed. But the positive class here is
a handful of events per fold, so a val block routinely holds one class and
`safe_auc` returns NaN for it -- and an epoch line reporting only NaN says
nothing about whether the model moved at all. The loss is finite there, which is
precisely the epoch where a readable number is wanted.

The loss also has to be on the same scale as the training objective, or it
cannot be read against it: `pos_weight` at this positive rate is ~200, so an
unweighted BCE would report a number two orders of magnitude smaller and look
like a model that had already converged.
"""
import numpy as np
import pytest
import torch

from waveform_forecast.data import MultiStationWindows, fit_stats
from waveform_forecast.metrics import safe_auc
from waveform_forecast.train import train_one_seed


class Args:
    """The knobs `train_one_seed` reads, at their smallest useful values."""
    arm = "features"
    hidden = 8
    proj_dim = None
    dropout = 0.0
    epochs = 1
    batch_size = 4
    lr = 1e-3
    weight_decay = 0.0
    patience = 1
    num_workers = 0
    seq_hours = 3


def windows(labels, n=20, seq_hours=3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2, 3)).astype(np.float32)
    present = np.ones((n, 2), dtype=bool)
    idx = np.arange(seq_hours - 1, n)
    stats = fit_stats(x, present, idx, seq_hours)
    return MultiStationWindows(x, present, np.asarray(labels, dtype=np.int64),
                               seq_hours, idx, stats)


def run(labels_train, labels_val, capsys, **kw):
    args = Args()
    for k, v in kw.items():
        setattr(args, k, v)
    tr, va, te = (windows(labels_train), windows(labels_val), windows(labels_train))
    train_one_seed(args, 42, tr, va, te, feat_dim=3, device=torch.device("cpu"))
    return capsys.readouterr().out


def test_the_epoch_line_reports_val_loss(capsys):
    labels = [0] * 15 + [1] * 5
    out = run(labels, labels, capsys)
    line = next(l for l in out.splitlines() if "epoch 1/1" in l)
    assert "val loss" in line
    value = float(line.split("val loss")[1].split()[0])
    assert np.isfinite(value) and value > 0


def test_train_loss_is_reported_beside_it(capsys):
    """A val loss alone cannot separate a model learning slowly from one that
    has begun memorising."""
    labels = [0] * 15 + [1] * 5
    line = next(l for l in run(labels, labels, capsys).splitlines()
                if "epoch 1/1" in l)
    assert "train loss" in line
    assert np.isfinite(float(line.split("train loss")[1].split()[0]))


def test_val_loss_is_finite_on_the_epoch_where_auc_is_nan(capsys):
    """A val block of one class. This is the ordinary case at this positive
    rate, not an edge case -- and the whole reason the loss is printed."""
    one_class = [0] * 20
    out = run([0] * 15 + [1] * 5, one_class, capsys)
    line = next(l for l in out.splitlines() if "epoch 1/1" in l)
    assert "nan" in line.split("val AUC")[1].lower(), "the AUC is undefined here"
    assert np.isfinite(float(line.split("val loss")[1].split()[0])), (
        "the loss must still be readable on exactly the epoch AUC cannot be")


def test_the_reported_loss_is_the_weighted_objective_not_a_plain_bce(capsys):
    """`pos_weight` at a 25% positive rate is 3; at the real rate it is ~200.
    Reporting an unweighted BCE would put the val loss on a different scale from
    the thing being minimised and make it unreadable against the train loss."""
    labels = [0] * 15 + [1] * 5
    line = next(l for l in run(labels, labels, capsys).splitlines()
                if "epoch 1/1" in l)
    val = float(line.split("val loss")[1].split()[0])
    # An untrained model sits near logit 0, where unweighted BCE is ln 2 ~ 0.69.
    # The weighted objective at this rate is meaningfully above that.
    assert val > 0.69, f"val loss {val:.4f} looks like an unweighted BCE"


def test_a_short_final_batch_does_not_dominate_the_mean(capsys):
    """19 windows at batch size 4 leaves a final batch of three. Averaging over
    batches rather than samples lets those three count as much as four.

    `epochs=0` is what makes this a controlled comparison: no optimiser steps
    run, so both calls score the identically-seeded initial weights and the
    ONLY difference is how the windows were grouped. Training first would
    change the step count with the batch size, and the two losses would differ
    for a reason that has nothing to do with the mean.
    """
    labels = [0] * 16 + [1] * 5
    a = run(labels, labels, capsys, batch_size=4, epochs=0)
    b = run(labels, labels, capsys, batch_size=64, epochs=0)
    la = float(next(l for l in a.splitlines() if "test loss" in l)
               .split("test loss")[1].split()[0])
    lb = float(next(l for l in b.splitlines() if "test loss" in l)
               .split("test loss")[1].split()[0])
    assert abs(la - lb) < 1e-5, (
        f"loss moved with batch size ({la:.6f} vs {lb:.6f}); the mean is over "
        f"batches, not over samples")


def test_the_test_line_reports_its_loss_too(capsys):
    labels = [0] * 15 + [1] * 5
    out = run(labels, labels, capsys)
    line = next(l for l in out.splitlines() if "test AUC" in l)
    assert "test loss" in line
    assert np.isfinite(float(line.split("test loss")[1].split()[0]))
