"""kNN conversion in WavLM layer-6 space and the HLTCOE variations (paper §2.1, §3.1).

Notation: query x[T, D], matching set S[N, D] of one target speaker.

- `knn_regress`: y_t = mean of the k nearest s_j by cosine distance (k = 4, uniform weights).
- multi-set pooling: y_t = sum_m w_m * y_t^(m) over M matching sets, fixed weights per utterance.
- `stretch`: length perturbation by linear interpolation of the query sequence in time
  (r ~ U(0.8, 1.2)); pitch is untouched because the vocoder sees the same frame rate.
- `uniform_noise`: eps ~ U([-delta, delta]^D) per frame, so ||eps||_inf < delta.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def knn_regress(query: torch.Tensor, matching: torch.Tensor, k: int = 4,
                synth: torch.Tensor | None = None, chunk: int = 2048) -> torch.Tensor:
    """For each query frame, average the `synth` rows of its k nearest `matching` rows (cosine).

    `synth` defaults to `matching`. Queries are processed in chunks of `chunk` frames so that the
    similarity matrix stays at chunk x N.
    """
    if synth is None:
        synth = matching
    m_hat = F.normalize(matching.float(), dim=-1)
    out = []
    for start in range(0, query.shape[0], chunk):
        q_hat = F.normalize(query[start:start + chunk].float(), dim=-1)
        sim = q_hat @ m_hat.T  # cosine distance = 1 - sim, so the k nearest are the k largest sims
        idx = sim.topk(k, dim=-1).indices  # [chunk, k]
        out.append(synth[idx].float().mean(dim=1))
    return torch.cat(out, dim=0)


def knn_pooled(query: torch.Tensor, matching_sets: list[torch.Tensor], weights: torch.Tensor | None = None,
               k: int = 4, synth_sets: list[torch.Tensor] | None = None) -> torch.Tensor:
    """Frame-level weighted average of kNN outputs over several matching sets (paper Fig. 1)."""
    if weights is None:
        weights = torch.full((len(matching_sets),), 1.0 / len(matching_sets))
    synth_sets = synth_sets or [None] * len(matching_sets)
    y = 0.0
    for w, s, syn in zip(weights.tolist(), matching_sets, synth_sets):
        y = y + w * knn_regress(query, s, k=k, synth=syn)
    return y


def stretched_frames(num_samples: int, ratio: float, sr: int = 16000, hop: int = 320,
                     min_seconds: float = 0.5) -> int:
    """Frame count after length perturbation: floor(max(r * D, 0.5 s) * sr / hop)."""
    seconds = max(ratio * num_samples / sr, min_seconds)
    return max(1, int(seconds * sr / hop))


def stretch(feats: torch.Tensor, frames: int) -> torch.Tensor:
    """Linear interpolation of [T, D] features to [frames, D]."""
    if frames == feats.shape[0]:
        return feats
    return F.interpolate(feats.T[None].float(), size=frames, mode="linear", align_corners=False)[0].T


def uniform_noise(shape, delta: float, generator: torch.Generator, device) -> torch.Tensor:
    """eps ~ U(-delta, delta), element-wise, drawn on `device` from `generator`."""
    u = torch.rand(shape, generator=generator, device=device)
    return (2.0 * u - 1.0) * delta
