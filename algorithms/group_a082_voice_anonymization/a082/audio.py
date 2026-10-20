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


def loudness_normalize(wav: torch.Tensor, sr: int = SR, target_lufs: float = -16.0) -> torch.Tensor:
    """Scale to `target_lufs` (ITU-R BS.1770, as kNN-VC does). Left unchanged when loudness is undefined
    (shorter than one 400 ms gating block, or silent)."""
    x = wav if wav.dim() == 2 else wav[None]
    if x.shape[-1] < int(0.4 * sr):
        return wav
    dev = "cuda" if torch.cuda.is_available() else "cpu"   # torchaudio's lfilter takes seconds on the CPU
    lufs = torchaudio.functional.loudness(x.to(dev), sr).item()
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
