# -*- coding: utf-8 -*-
"""性能指标（论文公式 7/8 与模型分类指标）。

SNR  : 不可感知性（>20 dB 视为可接受，[38]）
BER  : 鲁棒性（<20% 视为有效，[48]）
ODG  : 客观差异等级（> -1 视为感知不可察觉，[47]；需 peaqb 包）
"""
from __future__ import annotations

import numpy as np


def snr(orig: np.ndarray, proc: np.ndarray) -> float:
    """SNR = 10*lg( sum(A^2) / sum((A_tilde-A)^2) )，公式 7。"""
    orig = np.asarray(orig, dtype=np.float64)
    proc = np.asarray(proc, dtype=np.float64)
    n = min(orig.size, proc.size)
    if n == 0:
        return float("nan")
    a = orig[:n]
    b = proc[:n]
    err = np.sum((b - a) ** 2)
    if err < 1e-20:
        return float("inf")
    return float(10.0 * np.log10(np.sum(a ** 2) / err))


def ber(w: np.ndarray, w_hat: np.ndarray) -> float:
    """BER = sum(w xor w_hat) / n，公式 8。长度不同时对齐到公共长度。"""
    w = np.asarray(w, dtype=np.int64).ravel()
    wh = np.asarray(w_hat, dtype=np.int64).ravel()
    n = min(w.size, wh.size)
    if n == 0:
        return 1.0
    return float(np.mean(w[:n] != wh[:n]))


def accuracy(w, w_hat) -> float:
    return 1.0 - ber(w, w_hat)


def odg(orig: np.ndarray, proc: np.ndarray, sr: int = 44100) -> float | None:
    """客观差异等级（PEAQ 简化实现，需 pip install peaqb）。

    不可用时返回 None（报告/README 中注明该指标为可选）。
    """
    try:
        import peaqb
    except ImportError:
        return None
    orig = np.asarray(orig, dtype=np.float64)
    proc = np.asarray(proc, dtype=np.float64)
    n = min(orig.size, proc.size)
    return float(peaqb.calculate(orig[:n].astype(np.float32),
                                 proc[:n].astype(np.float32),
                                 sr))


def metrics_summary(orig_audio, wm_audio, w, w_hat, sr=44100, with_odg=False):
    out = {
        "snr_db": snr(orig_audio, wm_audio),
        "ber": ber(w, w_hat),
        "accuracy": accuracy(w, w_hat),
    }
    if with_odg:
        out["odg"] = odg(orig_audio, wm_audio, sr)
    return out
