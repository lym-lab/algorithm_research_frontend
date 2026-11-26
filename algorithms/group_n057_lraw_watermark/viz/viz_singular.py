# -*- coding: utf-8 -*-
"""图 4：奇异值稳定性（论文 Fig. 4 复刻）——同一音频在不同
“重录距离”（数字信道模型）下 5 个奇异值的变化。

预期结论：随距离增大奇异值整体下降，但 s1>s2>s3>s4>s5 的降序
始终保持 —— 这是 LRAW 嵌入/提取设计的事实基础。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lraw import (Config, DEFAULT_CONFIG, synth_music, synth_speech_like,
                  apply_attack, extract_features, vad)

FIG = Path(__file__).resolve().parent.parent / "output" / "figs"
DISTANCES = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 13.0]


def singular_at(audio, sr, seg_len=400, distance=1.0, cfg=None):
    cfg = cfg or DEFAULT_CONFIG
    y = apply_attack("recapture", audio, sr, cfg, distance_m=distance)
    voiced = vad(y, sr)
    n = voiced.size // seg_len
    segs = voiced[: n * seg_len].reshape(n, seg_len)
    mid = n // 3
    f = extract_features(segs[mid], cfg.wavelet, cfg.dwt_levels,
                         cfg.svd_rows, cfg.svd_cols)
    return f.S


def main():
    cfg = DEFAULT_CONFIG
    sr = cfg.sr
    FIG.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    samples = [
        ("synth music (seg 1/3)", synth_music(16.0, sr, seed=42)),
        ("synth speech (seg 1/3)", synth_speech_like(12.0, sr, seed=7)),
    ]
    colors = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed"]
    for ax, (label, audio) in zip(axes, samples):
        s0 = singular_at(audio, sr, distance=0.5)
        for i in range(5):
            vals = [singular_at(audio, sr, distance=d)[i] for d in DISTANCES]
            ax.plot(DISTANCES, vals, "o-", color=colors[i], label=f"s{i+1}",
                    lw=1.6, ms=5)
        ax.set_xscale("log")
        ax.set_xticks(DISTANCES)
        ax.set_xticklabels([f"{d:g}" for d in DISTANCES])
        ax.set_xlabel("Recapture distance (m)")
        ax.set_ylabel("Singular value")
        ax.set_title(f"{label}")
        ax.grid(alpha=0.3)
        ax.legend(ncol=5, fontsize=9)

    fig.suptitle("Stability of the five singular values under (simulated) "
                 "recapturing attacks at different distances", fontsize=12)
    fig.tight_layout()
    out = FIG / "fig_singular_stability.png"
    fig.savefig(out, dpi=160)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
