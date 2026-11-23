"""The scipy loudness agrees with torchaudio.functional.loudness (short signals: torchaudio is slow on the CPU)."""
import math

import torch
import torchaudio

from a082 import audio


def speechlike(seconds: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    n = int(16000 * seconds)
    t = torch.arange(n) / 16000
    env = (torch.sin(2 * math.pi * 3 * t) > 0).float() * 0.8 + 0.2           # syllable-like on/off
    return (0.3 * env * torch.randn(n, generator=g) + 0.2 * torch.sin(2 * math.pi * 180 * t))[None]


def test_matches_torchaudio():
    for seed, sec, scale in ((0, 1.3, 1.0), (1, 2.0, 0.05), (2, 3.1, 3.0)):   # scale 3: clamping in the filters
        x = speechlike(sec, seed) * scale
        ref = torchaudio.functional.loudness(x, 16000).item()
        assert abs(audio.loudness(x) - ref) < 1e-3, (seed, audio.loudness(x), ref)


def test_undefined_and_normalize():
    assert math.isnan(audio.loudness(torch.zeros(1, 1000)))
    x = speechlike(2.0, 3) * 0.2          # no sample near 1: the filters' clamping does not act
    y = audio.loudness_normalize(x)
    assert abs(audio.loudness(y) + 16.0) < 0.01
    silent = torch.zeros(1, 16000)
    assert torch.equal(audio.loudness_normalize(silent), silent)
