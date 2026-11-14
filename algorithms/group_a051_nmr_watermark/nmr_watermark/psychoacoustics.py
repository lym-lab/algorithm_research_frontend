"""Paper equations (2)-(3), with BS.1387 FFT basic-model masking.

Independent implementation of published equations described by Kabal (2002).
Adaptation: 44.1 kHz, 1024-point symmetric Hann, 512 hop; full-scale 1019.5 Hz
sine represents 92 dB SPL. Analytic Bark bands avoid rounded-table overlaps.
Internal noise and zero-initialized time smoothing follow the reference model.
"""
import numpy as np
import torch
from torch import nn
from .model import SAMPLE_RATE, N_FFT, HOP


def constants():
    dz = 0.25
    low, high = 7 * np.arcsinh(np.array([80., 18000.]) / 650)
    count = int(np.ceil((high - low) / dz))
    zl = low + np.arange(count) * dz
    zu = np.minimum(zl + dz, high)
    fl, fc, fu = [650 * np.sinh(z / 7) for z in (zl, (zl + zu) / 2, zu)]
    f = np.arange(N_FFT // 2 + 1) * SAMPLE_RATE / N_FFT
    df = SAMPLE_RATE / N_FFT
    mapping = np.maximum(0, np.minimum(fu[:, None], f[None, :] + df / 2)
                         - np.maximum(fl[:, None], f[None, :] - df / 2)) / df
    khz = np.maximum(f / 1000, 1e-12)
    w2 = 10 ** ((-2.184 * khz**-0.8 + 6.5 * np.exp(-0.6 * (khz - 3.3)**2)
                 - 0.001 * khz**3.6) / 10)
    distance = abs(1019.5 / SAMPLE_RATE * N_FFT - round(1019.5 / SAMPLE_RATE * N_FFT))
    d = distance * (N_FFT - 1) / N_FFT
    peak = np.sinc(d) / (1 - d**2)
    gain = 10**(92 / 20) / (peak * (N_FFT - 1) / 4)
    internal = 10 ** (1.456 * (fc / 1000)**-0.8 / 10)
    temporal = np.exp(-HOP / SAMPLE_RATE / (0.008 + 100 / fc * (0.030 - 0.008)))
    offset = 10 ** (-np.maximum(3, 0.25 * np.arange(count) * dz) / 10)
    return mapping, w2 * gain**2, fc, internal, temporal, offset


class NMRLoss(nn.Module):
    def __init__(self, normalized=False):
        super().__init__()
        for name, value in zip(("mapping", "ear_gain", "centers", "internal", "temporal", "offset"), constants()):
            self.register_buffer(name, torch.tensor(value, dtype=torch.float64))
        if normalized:
            self.ear_gain.mul_(N_FFT)
        bands = len(self.centers)
        source = torch.arange(bands)[:, None]
        destination = torch.arange(bands)[None, :]
        self.register_buffer("distance", destination - source)
        self.register_buffer("normalization", self._spread_raw(torch.ones(1, bands, dtype=torch.float64)))

    def _spread_raw(self, energy):
        # energy [..., source band]; sum nonlinear contributions to destination.
        count = energy.shape[-1]
        a_lower = 10**(-2.7 * 0.25)
        a_upper = 10 ** ((-2.4 - 23 / self.centers.to(energy)) * 0.25) * energy.clamp_min(1e-12)**0.05
        index = torch.arange(count, device=energy.device, dtype=energy.dtype)
        lower_sum = (1 - a_lower ** (index + 1)) / (1 - a_lower)
        # Finite geometric sum is stable also at a_upper == 1.
        powers = torch.arange(count, device=energy.device, dtype=energy.dtype)
        upper_sum = (a_upper[..., :, None] ** powers * (powers < (count - index)[:, None])).sum(-1)
        normalized = energy / (lower_sum + upper_sum - 1)
        distance = self.distance.to(energy)
        weights = torch.where(distance >= 0,
                              a_upper[..., :, None] ** (distance.clamp_min(0) * 0.4),
                              a_lower ** ((-distance).clamp_min(0) * 0.4))
        return (normalized.clamp_min(1e-30)[..., :, None]**0.4 * weights).sum(-2)**2.5

    @torch.no_grad()
    def masking(self, host):
        energy = torch.einsum("cf,bft->btc", self.mapping.to(host.real),
                              host.abs().square() * self.ear_gain.to(host.real)[None, :, None])
        energy = energy.clamp_min(1e-12) + self.internal.to(energy)
        flat = energy.reshape(-1, energy.shape[-1])
        spread = torch.cat([self._spread_raw(part) / self.normalization.to(part)
                            for part in flat.split(128)], dim=0).reshape_as(energy)
        state = torch.zeros_like(spread[:, 0])
        a = self.temporal.to(spread)
        smoothed = []
        for frame in spread.unbind(1):
            state = a * state + (1 - a) * frame
            smoothed.append(torch.maximum(state, frame))
        return (torch.stack(smoothed, dim=1) * self.offset.to(spread)).transpose(1, 2)

    def noise(self, marked, host):
        return torch.einsum("cf,bft->bct", self.mapping.to(host.real),
                            (marked - host).abs().square() * self.ear_gain.to(host.real)[None, :, None])

    def forward(self, marked, host, mask=None):
        if mask is None:
            mask = self.masking(host)
        return (self.noise(marked, host) / mask.clamp_min(1e-12)).mean(dim=(1, 2))
