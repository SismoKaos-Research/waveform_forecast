"""What an absent station must do to the answer: nothing at all.

The whole reason this project pools rather than concatenates is that the
stations do not share a span — MANT 756 days, DEMI 563, ELBA 252, GCAM 189,
with all three Aegean stations overlapping on 80. A model requiring every
station present would train on those 80 days and a couple of qualifying events.

Pooling only helps if absence is genuinely absence. Concatenation forces an
imputed value into the vector and zero-after-standardization reads as "exactly
average", which is indistinguishable from a real quiet reading. So the property
worth testing is not that the model runs with a mask, but that a masked-out
station contributes *exactly* nothing: with one station up the pooled embedding
IS that station's, to the bit, whatever garbage sits in the other slots.
"""
import pytest
import torch

from waveform_forecast.model import MaskedStationPool, MultiStationForecaster


@pytest.fixture
def pool():
    torch.manual_seed(0)
    return MaskedStationPool(8).eval()


@pytest.fixture
def emb():
    torch.manual_seed(1)
    return torch.randn(2, 3, 4, 8)          # batch 2, 3 hours, 4 stations, dim 8


def only(station, shape=(2, 3, 4)):
    m = torch.zeros(*shape, dtype=torch.bool)
    m[..., station] = True
    return m


# --- absence is absence ----------------------------------------------------

def test_one_station_present_pools_to_that_station_exactly(pool, emb):
    with torch.no_grad():
        pooled, _ = pool(emb, only(0))
    assert torch.equal(pooled, emb[..., 0, :])


def test_garbage_in_an_absent_slot_changes_nothing(pool, emb):
    """The concatenation failure, made impossible: an imputed value in a slot
    the mask says is empty must not reach the answer."""
    noisy = emb.clone()
    noisy[..., 1:, :] = 1e6
    with torch.no_grad():
        a, _ = pool(emb, only(0))
        b, _ = pool(noisy, only(0))
    assert torch.equal(a, b)


def test_absent_stations_take_no_weight(pool, emb):
    with torch.no_grad():
        _, w = pool(emb, only(2))
    assert float(w[..., 2].min()) == 1.0
    assert float(w[..., [0, 1, 3]].abs().max()) == 0.0


def test_weights_over_the_present_stations_sum_to_one(pool, emb):
    present = torch.zeros(2, 3, 4, dtype=torch.bool)
    present[..., :2] = True
    with torch.no_grad():
        _, w = pool(emb, present)
    assert torch.allclose(w.sum(-1), torch.ones(2, 3), atol=1e-6)


def test_which_station_is_present_actually_matters(pool, emb):
    """A pool that ignored the mask would pass every test above by averaging."""
    with torch.no_grad():
        a, _ = pool(emb, only(0))
        b, _ = pool(emb, only(1))
    assert not torch.allclose(a, b)


# --- the degenerate hour ---------------------------------------------------

def test_an_hour_with_no_station_is_zeros_not_nan(pool, emb):
    """softmax over an all -inf row is NaN, and one NaN poisons the whole
    backward pass. The hour is empty; it must say so numerically."""
    with torch.no_grad():
        pooled, w = pool(emb, torch.zeros(2, 3, 4, dtype=torch.bool))
    assert torch.isfinite(pooled).all()
    assert float(pooled.abs().max()) == 0.0
    assert float(w.abs().max()) == 0.0


def test_an_empty_hour_does_not_poison_its_neighbours(pool, emb):
    """One dead hour in a window must not NaN the hours around it."""
    present = only(0)
    present[:, 1, :] = False
    with torch.no_grad():
        pooled, _ = pool(emb, present)
    assert torch.isfinite(pooled).all()
    assert torch.equal(pooled[:, 0], emb[:, 0, 0, :])
    assert float(pooled[:, 1].abs().max()) == 0.0


def test_gradients_stay_finite_through_a_masked_pool(emb):
    """The real risk of masking with -inf: it trains fine and NaNs on backward."""
    p = MaskedStationPool(8)
    present = only(0)
    present[:, 2, :] = False                      # one empty hour in the window
    x = emb.clone().requires_grad_(True)
    pooled, _ = p(x, present)
    pooled.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(torch.isfinite(q.grad).all() for q in p.parameters() if q.grad is not None)


# --- the two arms ----------------------------------------------------------

def test_the_feature_arm_runs_and_returns_one_logit_per_window():
    torch.manual_seed(0)
    net = MultiStationForecaster(feat_dim=12, hidden=16).eval()
    x = torch.randn(3, 24, 2, 12)                 # 24 hours, 2 stations
    present = torch.ones(3, 24, 2, dtype=torch.bool)
    with torch.no_grad():
        out = net(x, present)
    assert out.shape == (3,)
    assert torch.isfinite(out).all()


def test_the_raw_arm_differs_by_one_argument():
    """Arm B is arm A with an encoder, exactly as SequenceHeadNet does it."""
    from waveform_forecast.blocks import RawWaveformEncoder
    torch.manual_seed(0)
    enc = RawWaveformEncoder(out_dim=16)
    net = MultiStationForecaster(feat_dim=16, hidden=16, encoder=enc).eval()
    x = torch.randn(2, 4, 2, 3, 900)              # 4 hours, 2 stations, 3 comp, samples
    present = torch.ones(2, 4, 2, dtype=torch.bool)
    with torch.no_grad():
        out = net(x, present)
    assert out.shape == (2,)
    assert torch.isfinite(out).all()


def test_the_model_reports_which_station_it_leaned_on():
    """Not decoration: with stations of very different coverage, a result that
    turns out to be one station's is a different claim than a fused one."""
    torch.manual_seed(0)
    net = MultiStationForecaster(feat_dim=12, hidden=16).eval()
    present = torch.ones(2, 6, 3, dtype=torch.bool)
    present[..., 2] = False
    with torch.no_grad():
        net(torch.randn(2, 6, 3, 12), present)
    assert net.last_weights.shape == (2, 6, 3)
    assert float(net.last_weights[..., 2].abs().max()) == 0.0
