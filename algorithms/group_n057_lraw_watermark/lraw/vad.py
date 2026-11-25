# -*- coding: utf-8 -*-
"""基于帧能量的 VAD（语音活动检测）。

论文引用 Hou et al. [35] 的语音增强方法做静音段剔除；本模块用一个
可配置的帧能量 VAD 近似，阈值可按分位数 / 均值 / 固定值三种方式设定。
"""
from __future__ import annotations

import numpy as np


def _frame_energy(x: np.ndarray, frame_len: int) -> np.ndarray:
    n_frames = x.size // frame_len
    if n_frames == 0:
        return np.array([], dtype=np.float64)
    frames = x[: n_frames * frame_len].reshape(n_frames, frame_len)
    return np.sqrt(np.mean(frames ** 2, axis=1))


def vad(x: np.ndarray,
        sr: int,
        frame_ms: float = 20.0,
        mode: str = "percentile",
        threshold: float = 0.15) -> np.ndarray:
    """返回去除静音后的语音/有声信号。

    参数
    ----
    x : 输入信号（float 一维）。
    mode : percentile（低于 threshold 分位数的帧判为静音，推荐）
         | mean（低于帧均能量的 threshold 倍判为静音）
         | fixed（低于绝对阈值 threshold 判为静音）
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    frame_len = max(1, int(round(sr * frame_ms / 1000.0)))
    energy = _frame_energy(x, frame_len)

    if energy.size == 0:
        return x.copy()

    if mode == "percentile":
        thr = np.quantile(energy, threshold)
    elif mode == "mean":
        thr = threshold * float(np.mean(energy))
    elif mode == "fixed":
        thr = float(threshold)
    else:
        raise ValueError(f"unknown vad mode: {mode}")

    keep = energy > thr
    # 简单的 hangover：把孤立静音帧前后 1 帧也判为有声，避免把词内停顿切碎
    if keep.size > 2:
        keep = np.convolve(keep.astype(np.int8), np.ones(3, dtype=np.int8), mode="same") > 0

    voiced = x[: keep.size * frame_len].reshape(keep.size, frame_len)[keep].ravel()
    return voiced
