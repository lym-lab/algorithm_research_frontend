"""Per-utterance randomness.

Every random choice for an utterance (target speaker, length factor, noise, TTS voice, admixture draw)
is derived from sha256(utt_id | key), so any utterance can be regenerated alone, in any order, on any
number of workers, and gets the same result.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch


def _digest(utt_id: str, key: str) -> bytes:
    return hashlib.sha256(f"{utt_id}|{key}".encode("utf-8")).digest()


def seed_int(utt_id: str, key: str) -> int:
    """63-bit seed for (utterance, purpose)."""
    return int.from_bytes(_digest(utt_id, key)[:8], "big") >> 1


def rng(utt_id: str, key: str) -> np.random.Generator:
    return np.random.default_rng(seed_int(utt_id, key))


def torch_gen(utt_id: str, key: str, device: str | torch.device = "cpu") -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed_int(utt_id, key))
    return g


def uniform01(utt_id: str, key: str) -> float:
    """A uniform number in [0, 1) fixed by (utterance, key); used for the admixture draw."""
    return seed_int(utt_id, key) / float(1 << 63)
