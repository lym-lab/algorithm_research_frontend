"""Training objectives of the WavLM conversion model.

- reconstruction: L1 + L2 between prediction and the kNN-VC target, before and after the PostNet;
- adversarial speaker loss (rows 6, 8): the converter maximises the speaker classifier's cross entropy,
  capped at log(n_speakers) so that "maximise CE" cannot diverge;
- discrete loss (rows 7, 8): CTC over k-means units of the target speaker, with logits
  sigma * cos(y_t, c_j) and a learnable blank vector.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean of x[B, T, D] over valid[B, T] frames and all dims."""
    w = valid[..., None].to(x.dtype)
    return (x * w).sum() / (w.sum() * x.shape[-1]).clamp_min(1.0)


def reconstruction(y: torch.Tensor, y_post: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    loss = 0.0
    for pred in (y, y_post):
        diff = pred - target
        loss = loss + masked_mean(diff.abs(), valid) + masked_mean(diff ** 2, valid)
    return loss


def capped_adversarial(logits: torch.Tensor, spk: torch.Tensor, n_speakers: int) -> torch.Tensor:
    """The term the converter subtracts: min(CE, log n)."""
    return torch.clamp(F.cross_entropy(logits, spk), max=math.log(n_speakers))


class CosineCTC(nn.Module):
    """CTC over K k-means centroids (fixed) plus one learnable blank; logits = sigma * cosine."""

    def __init__(self, centroids: torch.Tensor, sigma: float = 10.0):
        super().__init__()
        self.register_buffer("centroids", F.normalize(centroids.float(), dim=-1))
        self.blank = nn.Parameter(F.normalize(torch.randn(1, centroids.shape[1]), dim=-1))
        self.sigma = sigma

    def logits(self, y: torch.Tensor) -> torch.Tensor:
        """y[B, T, D] -> [B, T, K+1], index 0 = blank."""
        c = torch.cat([F.normalize(self.blank, dim=-1), self.centroids], dim=0)
        return self.sigma * F.normalize(y, dim=-1) @ c.T

    def forward(self, y: torch.Tensor, lengths: torch.Tensor, units: list[torch.Tensor]) -> torch.Tensor:
        """`units[b]` is the deduplicated unit sequence (values 0..K-1) of utterance b."""
        logp = self.logits(y).float().log_softmax(-1).transpose(0, 1)  # [T, B, K+1]
        tgt = torch.cat([u + 1 for u in units]).to(y.device)              # shift: 0 is blank
        tgt_len = torch.tensor([len(u) for u in units], device=y.device)
        return F.ctc_loss(logp, tgt, lengths, tgt_len, blank=0, reduction="mean", zero_infinity=True)


def dedup(units: torch.Tensor) -> torch.Tensor:
    """Merge consecutive repeats: [3, 3, 7, 7, 7, 3] -> [3, 7, 3]."""
    if units.numel() == 0:
        return units
    keep = torch.ones_like(units, dtype=torch.bool)
    keep[1:] = units[1:] != units[:-1]
    return units[keep]


def greedy_unit_error_rate(logits: torch.Tensor, lengths: torch.Tensor, units: list[torch.Tensor]) -> float:
    """CTC greedy decoding vs reference units; edit distance / reference length."""
    errs, total = 0, 0
    best = logits.argmax(-1)
    for b, ref in enumerate(units):
        hyp = dedup(best[b, : lengths[b]])
        hyp = hyp[hyp != 0] - 1
        errs += _edit(hyp.tolist(), ref.tolist())
        total += len(ref)
    return errs / max(total, 1)


def _edit(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]
