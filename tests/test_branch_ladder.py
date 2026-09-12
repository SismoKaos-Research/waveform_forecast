"""Spending less capacity on the hour axis, in four rungs.

The branch is the largest single component of the network -- 108,288 of the
feature arm's 279,241 parameters, ten times the raw CNN -- and the M>=4.5 label
is driven by about 30 qualifying events inside the archive span. Every run so
far has overfitted accordingly: train loss 0.58 -> 0.05 while val loss climbs
3.3 -> 10.3, with val AUC never leaving 0.46-0.50.

The one lever that has moved anything pointed the same direction: shortening the
context from 24 h to 8 h closed the gap to the floor from -0.050 to -0.003. Less
temporal modelling helped.

So the rungs go DOWN, and a cell swap is one of them rather than the whole
experiment:

    mean+linear   65,206     no temporal model, one Linear
    mean+mlp      78,375     no temporal model
    gru+mlp      244,297     3-gate cell
    lstm+mlp     279,241     the ported branch (the current default)

If the rungs agree, the 214k parameters between the ends are buying nothing and
the ceiling is the label, not the model. If the small ones win, every collapse
above a 0.6 floor was overfitting.
"""
import numpy as np
import pytest
import torch

from waveform_forecast.blocks import (BRANCHES, GRUAttentionBranch,
                                      LSTMAttentionBranch, MeanBranch)
from waveform_forecast.model import MultiStationForecaster


def params(m):
    return sum(p.numel() for p in m.parameters())


# --- the branches themselves -----------------------------------------------

def test_every_rung_is_reachable_by_name():
    assert set(BRANCHES) == {"mean", "gru", "lstm"}


def test_the_mean_branch_has_no_parameters_at_all():
    """The floor of the ladder is a mean, not a small network."""
    assert params(MeanBranch(16)) == 0


def test_the_mean_branch_is_the_mean_over_hours():
    b = MeanBranch(4)
    x = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    assert torch.allclose(b(x), x.mean(dim=1))


def test_the_mean_branch_does_not_change_the_width():
    assert MeanBranch(37).out_dim == 37


def test_the_recurrent_branches_double_the_width():
    """Bidirectional, so out_dim is hidden * 2 -- the head sizes off this."""
    assert GRUAttentionBranch(8, hidden=32).out_dim == 64
    assert LSTMAttentionBranch(8, hidden=32).out_dim == 64


def test_a_gru_is_smaller_than_an_lstm_by_about_a_gate():
    """3 gates against 4, everything else identical.

    The point of the rung: a cell swap buys ~10k parameters where the gap
    between the ends of the ladder is 214k. It is a 25% step, not a fix.
    """
    g, l = params(GRUAttentionBranch(16)), params(LSTMAttentionBranch(16))
    assert g < l
    assert 0.85 < g / l < 0.95


def test_the_gru_branch_differs_from_the_ported_one_only_in_the_cell():
    """Same attention, same residual, same LayerNorm, same time pooling."""
    g, l = GRUAttentionBranch(16), LSTMAttentionBranch(16)
    assert params(g.attn) == params(l.attn)
    assert params(g.norm) == params(l.norm)
    assert g.out_dim == l.out_dim


def test_the_ported_branch_is_untouched():
    """`LSTMAttentionBranch` is the thing the published figures came from.

    The ladder sits beside it; if this changes, a multi-station number stops
    being readable against the single-station ones.
    """
    assert BRANCHES["lstm"] is LSTMAttentionBranch
    assert isinstance(LSTMAttentionBranch(8).lstm, torch.nn.LSTM)


@pytest.mark.parametrize("name", ["mean", "gru", "lstm"])
def test_every_branch_pools_a_sequence_to_one_vector(name):
    b = BRANCHES[name](12, hidden=16)
    out = b(torch.randn(3, 7, 12))
    assert out.shape == (3, b.out_dim)
    assert torch.isfinite(out).all()


# --- the ladder as assembled models ----------------------------------------

RUNGS = [("mean", "linear"), ("mean", "mlp"), ("gru", "mlp"), ("lstm", "mlp")]


@pytest.mark.parametrize("branch,head", RUNGS)
def test_every_rung_runs_and_returns_one_logit_per_window(branch, head):
    m = MultiStationForecaster(feat_dim=6, hidden=16, branch=branch,
                               head=head).eval()
    x = torch.randn(2, 5, 3, 6)
    present = torch.ones(2, 5, 3, dtype=torch.bool)
    with torch.no_grad():
        out = m(x, present)
    assert out.shape == (2,)
    assert torch.isfinite(out).all()


def test_the_ladder_is_monotone_in_parameters():
    """Each rung is strictly smaller than the one above it."""
    counts = [params(MultiStationForecaster(feat_dim=207, hidden=64,
                                            branch=b, head=h))
              for b, h in RUNGS]
    assert counts == sorted(counts)
    assert counts[0] < counts[-1] / 3


def test_the_default_is_still_the_ported_architecture():
    """A run that names no rung must behave exactly as before the ladder."""
    a = MultiStationForecaster(feat_dim=9, hidden=16)
    b = MultiStationForecaster(feat_dim=9, hidden=16, branch="lstm", head="mlp")
    assert params(a) == params(b)
    assert isinstance(a.branch, LSTMAttentionBranch)


def test_a_linear_head_is_smaller_than_an_mlp_head():
    a = MultiStationForecaster(feat_dim=9, hidden=16, branch="mean",
                               head="linear")
    b = MultiStationForecaster(feat_dim=9, hidden=16, branch="mean", head="mlp")
    assert params(a) < params(b)


def test_an_unknown_rung_is_refused():
    with pytest.raises(ValueError):
        MultiStationForecaster(feat_dim=4, branch="transformer")
    with pytest.raises(ValueError):
        MultiStationForecaster(feat_dim=4, head="softmax")


@pytest.mark.parametrize("branch,head", RUNGS)
def test_the_mask_still_means_absence_on_every_rung(branch, head):
    """The pooling guarantee must not depend on which rung is chosen.

    With one station present the forecast is unchanged by garbage in the other
    slots -- the property the whole design rests on.
    """
    torch.manual_seed(0)
    m = MultiStationForecaster(feat_dim=6, hidden=16, branch=branch,
                               head=head).eval()
    x = torch.randn(2, 4, 3, 6)
    present = torch.zeros(2, 4, 3, dtype=torch.bool)
    present[..., 0] = True
    poisoned = x.clone()
    poisoned[..., 1:, :] = 1e6
    with torch.no_grad():
        a, b = m(x, present), m(poisoned, present)
    assert torch.allclose(a, b, atol=1e-4)


@pytest.mark.parametrize("branch,head", RUNGS)
def test_gradients_stay_finite_on_every_rung(branch, head):
    m = MultiStationForecaster(feat_dim=6, hidden=16, branch=branch, head=head)
    present = torch.zeros(2, 4, 3, dtype=torch.bool)
    present[..., 0] = True
    out = m(torch.randn(2, 4, 3, 6), present)
    out.sum().backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
