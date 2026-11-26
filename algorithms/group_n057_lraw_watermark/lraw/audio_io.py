# -*- coding: utf-8 -*-
"""音频读写与基础信号处理工具。"""
from __future__ import annotations

import numpy as np
import soundfile as sf


def read_audio(path: str) -> tuple[np.ndarray, int]:
    """读取 wav/flac 等音频，返回 (float32 单声道信号, 采样率)。

    多声道信号取均值合并为单声道，符合论文“mono WAV, 44.1 kHz, 16 bit”设定。
    """
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data, sr


def write_audio(path, data: np.ndarray, sr: int) -> None:
    """写出 16 bit PCM WAV（与论文一致）。

    path 可为文件路径或 file-like 对象（BytesIO 需显式 format='WAV'）。
    """
    clip = np.clip(data, -1.0, 1.0)
    if hasattr(path, "write"):
        sf.write(path, clip.astype(np.float32), sr, format="WAV",
                 subtype="PCM_16")
    else:
        sf.write(path, clip.astype(np.float32), sr, subtype="PCM_16")


def to_int16(data: np.ndarray) -> np.ndarray:
    return np.clip(np.round(data * 32767.0), -32768, 32767).astype(np.int16)


def from_int16(data: np.ndarray) -> np.ndarray:
    return data.astype(np.float64) / 32768.0


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x ** 2)))
