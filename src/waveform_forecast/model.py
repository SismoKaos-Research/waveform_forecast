"""Several stations into one forecast: encode each, pool over the ones that are up.

Not a runnable script -- imported only.

**Why pooling with a mask rather than concatenating station vectors.** Measured
from the AFAD archives actually on disk:

    MANT 756 days   DEMI 563   ELBA 252   GCAM 189
    DEMI+MANT 479 overlapping days
    all three Aegean stations:  80 days
    all four:                   29 days

A model that requires every station present therefore trains on 80 days -- of
order two qualifying events, which is the effective-sample-size trap this
project has documented more than once. Pooling over a presence mask lets the
model train on the union instead, using whichever stations are up in a given
hour, so the 479-day DEMI+MANT overlap and MANT's 756 solo days are all usable.

**A station that is down must not read as a station that is quiet.** Concatenation
forces an imputed value into the vector, and zero after standardization means
"exactly average", which is a reading the model cannot distinguish from a real
one. Here an absent station contributes nothing to the pool at all: with one
station present the pooled embedding IS that station's embedding, exactly. That
property is tested rather than asserted.

The two arms differ in one argument. Arm A passes `encoder=None` -- an hourly
feature vector is already an embedding. Arm B passes `RawWaveformEncoder`, a 1D
CNN over one hour of 5 Hz samples. This mirrors `SequenceHeadNet` in
`cnn_earthquake/src/sismokaos/model/sequence.py`, which does the same thing for
the single-station models.
"""
import torch
import torch.nn as nn

from waveform_forecast.blocks import LSTMAttentionBranch


class MaskedStationPool(nn.Module):
    """Attention-pools station embeddings, ignoring the stations that are absent.

    Scores each present station, softmaxes over the present ones only, and
    returns their weighted mean. An hour with no station at all returns zeros
    and is reported through `any_present` so the caller can drop it rather than
    train on a fabricated input.
    """

    def __init__(self, dim, hidden=None):
        """Initializes the scoring MLP.

        Args:
            dim: Width of each station's embedding.
            hidden: Hidden width of the scorer. Defaults to `max(8, dim // 2)`.
        """
        super().__init__()
        hidden = hidden or max(8, dim // 2)
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                   nn.Linear(hidden, 1))

    def forward(self, emb, present):
        """Pools over the station axis.

        Args:
            emb: Station embeddings, shape (batch, time, stations, dim).
            present: Boolean mask, shape (batch, time, stations). True where
                that station has real data for that hour.

        Returns:
            Tuple of (pooled, weights):
            `pooled` has shape (batch, time, dim); `weights` has shape
            (batch, time, stations) and is zero at absent stations.
        """
        s = self.score(emb).squeeze(-1)                       # (B, T, S)
        # -inf before the softmax, not zero after it: zeroing afterwards would
        # still let an absent station take probability mass from the present
        # ones and shrink the pooled vector toward the origin.
        s = s.masked_fill(~present, float("-inf"))
        any_present = present.any(dim=-1, keepdim=True)
        # A row with nothing present is all -inf, and softmax of that is NaN.
        # Give it a uniform row, then zero the result -- the caller is told via
        # the mask that the hour is empty.
        s = torch.where(any_present, s, torch.zeros_like(s))
        w = torch.softmax(s, dim=-1) * present.float()
        pooled = (emb * w.unsqueeze(-1)).sum(dim=-2)
        return pooled * any_present.float(), w


class MultiStationForecaster(nn.Module):
    """Per-station encoder -> masked pool over stations -> LSTM over hours -> logit."""

    def __init__(self, feat_dim, hidden=64, dropout=0.3, encoder=None,
                 proj_dim=None):
        """Initializes the encoder, pool, recurrent branch and head.

        Args:
            feat_dim: Per-station, per-hour width the pool sees -- the feature
                vector's width when `encoder` is None, or the encoder's
                `out_dim` when it is not.
            hidden: LSTM hidden size per direction, and head hidden width.
            dropout: Dropout used throughout.
            encoder: Optional per-(hour, station) encoder. None passes the
                feature vector through (arm A); a `RawWaveformEncoder` embeds
                one hour of raw samples (arm B).
            proj_dim: Width to project each station into before pooling.
                Defaults to `feat_dim`. Giving the pool a learned projection
                matters when stations differ in scale, which they do -- site
                response is a property of the station, not the earthquake.
        """
        super().__init__()
        self.encoder = encoder
        dim = proj_dim or feat_dim
        self.project = nn.Sequential(nn.Linear(feat_dim, dim), nn.GELU(),
                                     nn.Dropout(dropout))
        self.pool = MaskedStationPool(dim)
        self.branch = LSTMAttentionBranch(dim, hidden=hidden, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(self.branch.out_dim),
            nn.Dropout(dropout),
            nn.Linear(self.branch.out_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.last_weights = None

    def forward(self, x, present):
        """Forecasts from one batch of multi-station windows.

        Args:
            x: Input. Shape (batch, time, stations, feat_dim) with no encoder;
                (batch, time, stations, channels, samples) with one.
            present: Boolean mask, shape (batch, time, stations).

        Returns:
            Tensor of shape (batch,) -- one raw logit per window. The pooling
            weights are kept on `self.last_weights` so a run can report which
            station the model actually leaned on.
        """
        if self.encoder is not None:
            b, t, s = x.shape[:3]
            x = self.encoder(x.reshape(b * t * s, *x.shape[3:])).reshape(b, t, s, -1)
        emb = self.project(x)
        pooled, self.last_weights = self.pool(emb, present)
        return self.head(self.branch(pooled)).squeeze(-1)
