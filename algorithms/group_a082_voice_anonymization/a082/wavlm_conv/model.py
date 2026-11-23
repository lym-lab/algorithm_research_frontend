"""WavLM conversion model (Table 1 rows 5-8; paper §2.2).

A FastSpeech2-style feed-forward Transformer that maps WavLM layer-6 features of any speaker to the
features kNN-VC would produce for one fixed target speaker, frame by frame:

    Linear 1024->512 -> +PE -> FFT x 4 (encoder) -> +PE -> FFT x 6 (decoder) -> Linear 512->1024 -> (+PostNet)

FFT block as in ming024/FastSpeech2 (the implementation cited by the paper through Chien et al.):
multi-head self-attention and a two-layer 1-D convolutional feed-forward net (kernels 9 and 1),
each followed by dropout, residual connection and post-LayerNorm. No variance adaptor or length
regulator: input and output are aligned frame by frame.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoid_table(n: int, d: int) -> torch.Tensor:
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    i = torch.arange(d, dtype=torch.float32)[None]
    angle = pos / torch.pow(10000.0, 2 * torch.div(i, 2, rounding_mode="floor") / d)
    table = torch.zeros(n, d)
    table[:, 0::2] = torch.sin(angle[:, 0::2])
    table[:, 1::2] = torch.cos(angle[:, 1::2])
    return table


class FFTBlock(nn.Module):
    def __init__(self, d: int = 512, heads: int = 2, filt: int = 1024, kernels=(9, 1), dropout: float = 0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.conv1 = nn.Conv1d(d, filt, kernels[0], padding=(kernels[0] - 1) // 2)
        self.conv2 = nn.Conv1d(filt, d, kernels[1], padding=(kernels[1] - 1) // 2)
        self.norm2 = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad: torch.Tensor | None) -> torch.Tensor:
        """x[B, T, d]; pad[B, T] is True at padded frames."""
        a, _ = self.attn(x, x, x, key_padding_mask=pad, need_weights=False)
        x = self.norm1(x + self.drop(a))
        if pad is not None:
            x = x.masked_fill(pad[..., None], 0.0)
        h = self.conv2(torch.relu(self.conv1(x.transpose(1, 2)))).transpose(1, 2)
        x = self.norm2(x + self.drop(h))
        if pad is not None:
            x = x.masked_fill(pad[..., None], 0.0)
        return x


class PostNet(nn.Module):
    """Five 1-D conv layers with batch norm (tanh on all but the last), predicting a residual."""

    def __init__(self, dim: int = 1024, channels: int = 512, kernel: int = 5, layers: int = 5, dropout: float = 0.5):
        super().__init__()
        mods = []
        for i in range(layers):
            cin = dim if i == 0 else channels
            cout = dim if i == layers - 1 else channels
            mods.append(nn.Sequential(nn.Conv1d(cin, cout, kernel, padding=(kernel - 1) // 2), nn.BatchNorm1d(cout)))
        self.layers = nn.ModuleList(mods)
        self.drop = nn.Dropout(dropout)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        h = y.transpose(1, 2)
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = torch.tanh(h)
            h = self.drop(h)
        return h.transpose(1, 2)


class WavLMConverter(nn.Module):
    def __init__(self, feat_dim: int = 1024, d: int = 512, enc_layers: int = 4, dec_layers: int = 6,
                 heads: int = 2, filt: int = 1024, dropout: float = 0.2, max_len: int = 4000, postnet: bool = True):
        super().__init__()
        self.inp = nn.Linear(feat_dim, d)
        self.register_buffer("pe", sinusoid_table(max_len, d), persistent=False)
        self.encoder = nn.ModuleList([FFTBlock(d, heads, filt, dropout=dropout) for _ in range(enc_layers)])
        self.decoder = nn.ModuleList([FFTBlock(d, heads, filt, dropout=dropout) for _ in range(dec_layers)])
        self.out = nn.Linear(d, feat_dim)
        self.postnet = PostNet(feat_dim) if postnet else None
        self.d = d

    def _pe(self, t: int) -> torch.Tensor:
        if t > self.pe.shape[0]:
            return sinusoid_table(t, self.d).to(self.pe.device)
        return self.pe[:t]

    def forward(self, x: torch.Tensor, pad: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """x[B, T, 1024] -> (before PostNet, after PostNet), both [B, T, 1024]."""
        h = self.inp(x) + self._pe(x.shape[1])[None]
        for blk in self.encoder:
            h = blk(h, pad)
        h = h + self._pe(x.shape[1])[None]
        for blk in self.decoder:
            h = blk(h, pad)
        y = self.out(h)
        y_post = y + self.postnet(y) if self.postnet is not None else y
        if pad is not None:
            y = y.masked_fill(pad[..., None], 0.0)
            y_post = y_post.masked_fill(pad[..., None], 0.0)
        return y, y_post


def lengths_to_pad(lengths: torch.Tensor, t: int | None = None) -> torch.Tensor:
    t = int(lengths.max()) if t is None else t
    return torch.arange(t, device=lengths.device)[None] >= lengths[:, None]


class NoamLR(torch.optim.lr_scheduler.LambdaLR):
    """lr = d^-0.5 * min(step^-0.5, step * warmup^-1.5) (ming024 default schedule, init_lr scaled).
    `offset` starts the schedule at that step instead of 0 (the variants continue the base's schedule)."""

    def __init__(self, opt, d: int = 512, warmup: int = 4000, last_epoch: int = -1, offset: int = 0):
        f = lambda s: (d ** -0.5) * min(max(s + offset, 1) ** -0.5, max(s + offset, 1) * warmup ** -1.5)
        super().__init__(opt, f, last_epoch)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


__all__ = ["WavLMConverter", "NoamLR", "lengths_to_pad", "count_params", "math"]
