"""Audio input/output shared by all anonymizers.

Every anonymizer reads utterances through `load_scp_entry` (Kaldi `wav.scp` values, including
`flac -c -d -s file.flac |` pipes as used by VPC 2024 for train-clean-360) and writes them with
`save_wav16`, so the 16 kHz / 16-bit PCM requirement of the VPC 2024 evaluation plan holds everywhere.
"""
from __future__ import annotations

import io
import math
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

SR = 16000


def load_scp_entry(entry: str, root: str | Path | None = None) -> tuple[torch.Tensor, int]:
    """Read one `wav.scp` value. Returns (wav[1, T] float32, sample rate).

    `entry` is either a path or a shell command ending in `|` whose stdout is an audio file.
    Relative paths and commands are resolved against `root` (the VPC repository root).
    """
    entry = entry.strip()
    if entry.endswith("|"):
        out = subprocess.run(entry[:-1], shell=True, cwd=root, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, check=True).stdout
        data, sr = sf.read(io.BytesIO(out), dtype="float32", always_2d=True)
    else:
        path = Path(entry)
        if root is not None and not path.is_absolute():
            path = Path(root) / path
        if path.suffix.lower() in (".m4a", ".aac", ".mp4"):   # VoxCeleb2: decoded by ffmpeg, rate unchanged
            out = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path), "-f", "wav", "-"],
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True).stdout
            data, sr = sf.read(io.BytesIO(out), dtype="float32", always_2d=True)
        else:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T.copy())
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav, sr


def load_audio(path: str | Path, sr: int = SR) -> torch.Tensor:
    """Read any file soundfile can read, as mono at `sr`. Returns wav[1, T]."""
    wav, file_sr = load_scp_entry(str(path))
    return resample(wav, file_sr, sr)


def resample(wav: torch.Tensor, sr_in: int, sr_out: int = SR) -> torch.Tensor:
    if sr_in == sr_out:
        return wav
    return torchaudio.functional.resample(wav, orig_freq=sr_in, new_freq=sr_out)


def _biquad_clamped(x: np.ndarray, b: list[float], a: list[float]) -> np.ndarray:
    from scipy.signal import lfilter
    return np.clip(lfilter(b, a, x, axis=-1), -1.0, 1.0)   # torchaudio's lfilter clamps by default


def loudness(wav: torch.Tensor | np.ndarray, sr: int = SR) -> float:
    """ITU-R BS.1770-4 loudness in LUFS of a [channels, time] or [time] signal: the same K-weighting biquads
    (with torchaudio's clamping after each), 400 ms blocks with 75 % overlap, absolute and relative gating as
    torchaudio.functional.loudness, computed with scipy's lfilter on the CPU. torchaudio's lfilter runs sample
    by sample: ~0.3 s per 10 s of audio on a V100 (2026-10-22), more than the HiFi-GAN vocoder itself, and
    seconds on a CPU; this takes milliseconds and agrees with it to float32 rounding (tests/test_audio_loudness)."""
    x = wav.detach().cpu().double().numpy() if isinstance(wav, torch.Tensor) else np.asarray(wav, dtype=np.float64)
    x = x[None] if x.ndim == 1 else x
    # treble shelf, gain 4 dB at 1500 Hz, Q 1/sqrt(2); then high-pass at 38 Hz, Q 0.5 (torchaudio's formulas)
    w0, Q, A = 2 * math.pi * 1500.0 / sr, 1 / math.sqrt(2), math.exp(4.0 / 40 * math.log(10))
    alpha = math.sin(w0) / 2 / Q
    t1, t2, t3 = 2 * math.sqrt(A) * alpha, (A - 1) * math.cos(w0), (A + 1) * math.cos(w0)
    x = _biquad_clamped(x, [A * ((A + 1) + t2 + t1), -2 * A * ((A - 1) + t3), A * ((A + 1) + t2 - t1)],
                        [(A + 1) - t2 + t1, 2 * ((A - 1) - t3), (A + 1) - t2 - t1])
    w0, Q = 2 * math.pi * 38.0 / sr, 0.5
    alpha, c = math.sin(w0) / 2 / Q, math.cos(w0)
    x = _biquad_clamped(x, [(1 + c) / 2, -1 - c, (1 + c) / 2], [1 + alpha, -2 * c, 1 - alpha])
    gate = int(round(0.4 * sr))
    step = int(round(gate * 0.25))
    n_blocks = (x.shape[-1] - gate) // step + 1
    if n_blocks < 1:
        return float("nan")
    sq = np.square(x)
    csum = np.concatenate([np.zeros((sq.shape[0], 1)), np.cumsum(sq, axis=-1)], axis=-1)
    starts = np.arange(n_blocks) * step
    energy = (csum[:, starts + gate] - csum[:, starts]) / gate                  # [channels, blocks]
    g = np.array([1.0, 1.0, 1.0, 1.41, 1.41])[: energy.shape[0], None]
    with np.errstate(divide="ignore", invalid="ignore"):
        block_lufs = -0.691 + 10 * np.log10(np.sum(g * energy, axis=0))
        keep = block_lufs > -70.0
        rel = -0.691 + 10 * np.log10(np.sum(g[:, 0] * energy[:, keep].mean(axis=-1))) - 10
        keep &= block_lufs > rel
        return float(-0.691 + 10 * np.log10(np.sum(g[:, 0] * energy[:, keep].mean(axis=-1))))


def loudness_normalize(wav: torch.Tensor, sr: int = SR, target_lufs: float = -16.0) -> torch.Tensor:
    """Scale to `target_lufs` (ITU-R BS.1770, as kNN-VC does). Left unchanged when loudness is undefined
    (shorter than one 400 ms gating block, or silent)."""
    x = wav if wav.dim() == 2 else wav[None]
    if x.shape[-1] < int(0.4 * sr):
        return wav
    lufs = loudness(x, sr)
    if not math.isfinite(lufs):
        return wav
    return torchaudio.functional.gain(wav, target_lufs - lufs)


def save_wav16(path: str | Path, wav: torch.Tensor | np.ndarray, sr: int = SR) -> None:
    """Write mono 16-bit PCM; samples are clipped to [-1, 1]."""
    if sr != SR:
        raise ValueError(f"outputs must be {SR} Hz, got {sr}")
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().float().numpy()
    wav = np.clip(np.asarray(wav, dtype=np.float32).reshape(-1), -1.0, 1.0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.wav")
    sf.write(tmp, wav, sr, subtype="PCM_16")
    tmp.replace(path)  # a crash never leaves a truncated file under the final name


def vad_trim(wav: torch.Tensor, sr: int = SR, trigger_level: float = 7.0) -> torch.Tensor:
    """Trim leading and trailing silence with sox-style VAD, applied from both ends
    (same procedure and default trigger level as kNN-VC's matching-set preparation)."""
    vad = torchaudio.transforms.Vad(sample_rate=sr, trigger_level=trigger_level)
    x = vad(wav)
    x = torch.flip(vad(torch.flip(x, (-1,))), (-1,))
    return x
