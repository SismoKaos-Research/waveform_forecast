"""The recurrent branch over hours, and the CNN that embeds one hour of samples.

Not a runnable script -- imported only.

`LSTMAttentionBranch` is unchanged from `cnn_earthquake/src/sismokaos/model/blocks.py`;
`RawWaveformEncoder` is unchanged from its `waveform.py`. Both produced the
published single-station forecasting figures, and keeping them identical is what
lets a multi-station result be read against those.
"""
import torch
import torch.nn as nn


class LSTMAttentionBranch(nn.Module):
    """LSTM for long-range order, then multi-head self-attention to weight steps."""

    def __init__(self, in_dim, hidden=64, layers=1, heads=4, dropout=0.2):
        """Initializes the LSTM+attention branch.

        Args:
            in_dim: Size of each step's input feature vector.
            hidden: LSTM hidden size per direction; the bidirectional output
                (and attention/LayerNorm width) is ``hidden * 2``.
            layers: Number of stacked LSTM layers.
            heads: Number of attention heads. Must divide ``hidden * 2``.
            dropout: Dropout used inside the LSTM (when ``layers > 1``) and
                the attention module.
        """
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        d = hidden * 2
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.out_dim = d

    def forward(self, x):
        """Encodes a sequence into one pooled embedding.

        Args:
            x: Input sequence, shape (batch, time, in_dim).

        Returns:
            Tensor of shape (batch, out_dim), the time-mean of the
            attention+residual output.
        """
        h, _ = self.lstm(x)
        a, _ = self.attn(h, h, h)
        h = self.norm(h + a)             # residual, as in the transformer block
        return h.mean(dim=1)             # pool over time

class RawWaveformEncoder(nn.Module):
    """1D CNN that embeds one hour's raw 3-component waveform."""

    def __init__(self, out_dim=32, dropout=0.3):
        super().__init__()

        def block(cin, cout, k, s):
            return nn.Sequential(nn.Conv1d(cin, cout, k, stride=s, padding=k // 2),
                                 nn.BatchNorm1d(cout), nn.GELU(), nn.Dropout(dropout))

        self.net = nn.Sequential(
            block(3, 16, 7, 4),
            block(16, 32, 5, 4),
            block(32, 32, 5, 4),
            block(32, out_dim, 3, 4),
            nn.AdaptiveAvgPool1d(1),
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x).squeeze(-1)



# --- the ablation ladder ---------------------------------------------------
#
# `LSTMAttentionBranch` above is a byte-identical port and stays that way; these
# sit beside it so a run can spend less capacity on the hour axis without
# touching the thing the published figures came from.
#
# The motivation is a parameter count against an event count. The branch is
# 116,480 of the feature arm's ~279,000 parameters -- the single largest
# component, larger than the raw CNN by 10x -- and the label at M>=4.5 is driven
# by about 30 qualifying events in the archive span. That is thousands of
# parameters per event, and every run so far has overfitted accordingly: train
# loss 0.58 -> 0.05 while val loss climbs 3.3 -> 10.3, with val AUC never
# leaving 0.46-0.50.
#
# The one lever that has moved anything pointed the same way -- shortening the
# context from 24 h to 8 h closed the gap to the floor from -0.050 to -0.003.
# Less temporal modelling helped. So the useful experiment is not a different
# recurrent cell at the same size, it is a ladder DOWN in capacity, with the
# cell swap as one rung of it.


class MeanBranch(nn.Module):
    """No temporal model at all: the mean over hours. Zero parameters.

    The floor of the ladder. If this matches the recurrent branches, then the
    116k parameters they spend on the hour axis are buying nothing, and the
    ceiling on this data is the label rather than the model.
    """

    def __init__(self, in_dim, **_):
        """Takes and ignores the recurrent branches' keyword arguments.

        Ignored rather than rejected so the four rungs are interchangeable at
        the call site and a ladder run differs by one string, not by a branch
        in the trainer.
        """
        super().__init__()
        self.out_dim = in_dim

    def forward(self, x):
        """(batch, time, in_dim) -> (batch, in_dim)."""
        return x.mean(dim=1)


class GRUAttentionBranch(nn.Module):
    """`LSTMAttentionBranch` with a GRU in place of the LSTM, and nothing else.

    Deliberately a one-line difference from the ported branch: a GRU has three
    gates where an LSTM has four, so this is ~3/4 of the recurrent parameters
    and otherwise the same architecture, the same attention, the same residual
    and the same pooling. Any difference in a paired run is therefore the cell,
    which is the only way that comparison means anything.
    """

    def __init__(self, in_dim, hidden=64, layers=1, heads=4, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        d = hidden * 2
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.out_dim = d

    def forward(self, x):
        """Encodes a sequence into one pooled embedding."""
        h, _ = self.gru(x)
        a, _ = self.attn(h, h, h)
        h = self.norm(h + a)             # residual, as in the transformer block
        return h.mean(dim=1)             # pool over time


BRANCHES = {
    "mean": MeanBranch,
    "gru": GRUAttentionBranch,
    "lstm": LSTMAttentionBranch,
}
