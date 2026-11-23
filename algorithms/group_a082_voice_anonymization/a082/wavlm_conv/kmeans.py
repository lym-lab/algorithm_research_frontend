"""Spherical k-means on the target speaker's WavLM frames (rows 7, 8; K = 100, cosine)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def spherical_kmeans(x: torch.Tensor, k: int = 100, iters: int = 50, seed: int = 0) -> torch.Tensor:
    """x[N, D] -> unit-norm centroids [k, D]. k-means++ initialisation on cosine similarity."""
    g = torch.Generator(device=x.device).manual_seed(seed)
    x = F.normalize(x.float(), dim=-1)
    n = x.shape[0]
    c = [x[torch.randint(n, (1,), generator=g, device=x.device)].squeeze(0)]
    dist = 1 - x @ c[0]
    for _ in range(1, k):
        p = dist.clamp_min(0) ** 2
        i = torch.multinomial(p / p.sum(), 1, generator=g)
        c.append(x[i].squeeze(0))
        dist = torch.minimum(dist, 1 - x @ c[-1])
    c = torch.stack(c)
    for _ in range(iters):
        assign = (x @ c.T).argmax(-1)
        new = torch.zeros_like(c).index_add_(0, assign, x)
        empty = new.norm(dim=-1) == 0
        new[empty] = c[empty]  # keep a centroid that lost all its points
        c = F.normalize(new, dim=-1)
    return c


def assign(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    return (F.normalize(x.float(), dim=-1) @ centroids.T).argmax(-1)
