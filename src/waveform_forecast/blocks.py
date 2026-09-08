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

