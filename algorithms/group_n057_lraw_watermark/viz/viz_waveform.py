# -*- coding: utf-8 -*-
"""图 2：波形对比（原始 / 含水印 / 攻击后）+ 局部放大。
图 3：频谱图对比（原始 vs 含水印）。

用法：
    python viz/viz_waveform.py --demo-meta output/demo_meta.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as sig

from lraw import synth_music, read_audio, write_audio

FIG = Path(__file__).resolve().parent.parent / "output" / "figs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=None, help="宿主 wav（缺省合成）")
    ap.add_argument("--wm-audio", default=None, help="含水印 wav")
    ap.add_argument("--atk-audio", default=None, help="攻击后 wav")
    args = ap.parse_args()

    sr = 44100
    if args.audio:
        orig, sr = read_audio(args.audio)
    else:
        orig = synth_music(8.0, sr)
    if args.wm_audio:
        wm, _ = read_audio(args.wm_audio)
    else:
        wm = orig.copy() + 0.01 * np.random.default_rng(0).standard_normal(orig.size)
    if args.atk_audio:
        atk, _ = read_audio(args.atk_audio)
    else:
        atk = wm.copy()

    FIG.mkdir(parents=True, exist_ok=True)
    t = np.arange(min(orig.size, wm.size, atk.size)) / sr
    n = t.size

    # ---- 波形 ----
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for ax, y, lab, c in zip(axes, [orig, wm, atk, wm - orig],
                             ["Original", "Watermarked", "Attacked",
                              "Watermark (Ã − A)"],
                             ["#2563eb", "#059669", "#dc2626", "#d97706"]):
        ax.plot(t, y[:n], color=c, lw=0.7)
        ax.set_ylabel(lab, fontsize=10)
        ax.set_xlim(0, t[-1])
    axes[0].set_title("Waveform comparison (synthetic music host)")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_waveform.png", dpi=160)
    print("[saved]", FIG / "fig_waveform.png")

    # ---- 局部放大（前 0.2 s）-----
    fig, ax = plt.subplots(figsize=(12, 3.5))
    k = int(0.2 * sr)
    ax.plot(t[:k], orig[:k], label="original", color="#2563eb", lw=0.8)
    ax.plot(t[:k], wm[:k], label="watermarked", color="#059669", lw=0.8,
            alpha=0.85)
    ax.set_xlabel("Time (s)")
    ax.set_title("Zoom-in: 0-0.2 s (embedding perturbation)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig_waveform_zoom.png", dpi=160)
    print("[saved]", FIG / "fig_waveform_zoom.png")

    # ---- 频谱图 ----
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    for ax, y, lab in zip(axes, [orig, wm], ["Original", "Watermarked"]):
        f, tt, Sxx = sig.spectrogram(y[:n], sr, nperseg=1024, noverlap=512)
        ax.pcolormesh(tt, f, 10 * np.log10(Sxx + 1e-12), shading="auto",
                      cmap="magma")
        ax.set_ylim(0, 12000)
        ax.set_ylabel("Freq (Hz)")
        ax.set_title(f"Spectrogram - {lab}")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_spectrogram.png", dpi=160)
    print("[saved]", FIG / "fig_spectrogram.png")


if __name__ == "__main__":
    main()
