# -*- coding: utf-8 -*-
"""合成测试音频（无真实语料时的可复现替代，论文用 FMA 音乐/LibriSpeech）。

demo 默认用合成“类音乐”信号：多谐波叠加 + 幅度调制 + 轻噪声，
保证 DWT-GBT-SVD 特征结构稳定且可复现。放入 data/ 的真实 wav 也可直接用。
"""
from __future__ import annotations

import numpy as np


def synth_music(duration_s: float = 12.0, sr: int = 44100, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # 由种子派生节奏/音区/噪声，制造宿主多样性（训练泛化用）
    tempo = rng.uniform(0.75, 1.35)          # 整体音高/速度偏移
    noise_db = 10 ** (rng.uniform(-2.8, -2.0) / 20)
    t = np.arange(int(duration_s * sr)) / sr
    x = np.zeros_like(t)
    # 三组基频（模仿和弦进行），各含 4 个谐波；每次生成随机转位
    roots = rng.uniform(0.9, 1.15, 4)
    chords = [
        (220.0 * roots[0] * tempo, 277.0 * roots[0] * tempo, 330.0 * roots[0] * tempo),
        (196.0 * roots[1] * tempo, 247.0 * roots[1] * tempo, 294.0 * roots[1] * tempo),
        (174.0 * roots[2] * tempo, 220.0 * roots[2] * tempo, 261.0 * roots[2] * tempo),
        (164.0 * roots[3] * tempo, 207.0 * roots[3] * tempo, 246.0 * roots[3] * tempo),
    ]
    seg = int(sr * 2.0)
    n_chords = int(np.ceil(t.size / seg))
    for c in range(n_chords):
        start = c * seg
        end = min(start + seg, t.size)
        tt = t[start:end] - t[start]
        for base in chords[c % len(chords)]:
            for h in range(1, 5):
                amp = 0.18 / h
                x[start:end] += amp * np.sin(2 * np.pi * base * h * tt)
    # 振幅包络（防爆音/削顶）
    env = np.minimum(1.0, t / 0.05) * np.minimum(1.0, (t.size - t) / 0.2)
    x *= env
    # 轻微泛音噪声（随种子变化）
    x += noise_db * rng.standard_normal(x.size)
    peak = np.max(np.abs(x))
    return (x / peak) * 0.9


def synth_speech_like(duration_s: float = 8.0, sr: int = 44100, seed: int = 7) -> np.ndarray:
    """类语音：浊音段（脉冲串激励经声道滤波）+ 清音段 + 停顿。

    f0 与浊/清/停顿分布参数由 seed 派生，制造宿主多样性。
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * sr)) / sr
    x = np.zeros_like(t)
    pos = 0
    f0 = rng.uniform(95.0, 165.0)
    p_voiced = rng.uniform(0.45, 0.65)
    p_unvoiced = rng.uniform(0.15, 0.28)
    while pos < x.size:
        # 随机浊音时长 0.15~0.5s / 清音 0.05~0.15s / 停顿 0.08~0.3s
        kind = rng.choice(["voiced", "unvoiced", "pause"],
                          p=[p_voiced, p_unvoiced, 1 - p_voiced - p_unvoiced])
        dur = rng.uniform(0.15, 0.5) if kind == "voiced" else \
              rng.uniform(0.05, 0.15) if kind == "unvoiced" else rng.uniform(0.08, 0.3)
        n = int(dur * sr)
        end = min(pos + n, x.size)
        seg_t = t[pos:end]
        if kind == "voiced":
            pulse = np.sign(np.sin(2 * np.pi * f0 * seg_t))
            # 简单声道共振（2 个共振峰带通近似；共振峰中心也随种子变化）
            from scipy.signal import butter, filtfilt
            fc1 = rng.uniform(350, 700) / (sr / 2)
            fc2 = rng.uniform(1800, 3200) / (sr / 2)
            b, a = butter(2, [fc1, fc2], btype="band")
            x[pos:end] = filtfilt(b, a, pulse) * 0.5
        elif kind == "unvoiced":
            x[pos:end] = 0.08 * rng.standard_normal(seg_t.size)
        pos = end
    peak = np.max(np.abs(x))
    return (x / peak) * 0.8
