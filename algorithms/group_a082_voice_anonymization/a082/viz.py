"""Figures shared by the web demo and the report: waveform and spectrogram of original vs anonymized speech."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

from .audio import SR


def log_mel(wav: torch.Tensor, sr: int = SR, n_mels: int = 80) -> np.ndarray:
    mel = torchaudio.transforms.MelSpectrogram(sr, n_fft=1024, win_length=1024, hop_length=160, n_mels=n_mels)(
        wav.reshape(1, -1).float())
    return torch.log10(mel.clamp_min(1e-5))[0].numpy()


def compare_figure(orig: torch.Tensor, anon: torch.Tensor, sr: int = SR,
                   titles=("Original", "Anonymized")) -> plt.Figure:
    """2 x 2 grid: waveform (top) and log-mel spectrogram (bottom) for each signal, same scales."""
    fig, ax = plt.subplots(2, 2, figsize=(11, 5), sharey="row", constrained_layout=True)
    specs = [log_mel(w, sr) for w in (orig, anon)]
    vmin, vmax = min(s.min() for s in specs), max(s.max() for s in specs)
    for j, (w, spec, title) in enumerate(zip((orig, anon), specs, titles)):
        x = w.reshape(-1).numpy()
        t = np.arange(len(x)) / sr
        ax[0, j].plot(t, x, lw=0.4, color="#2b6cb0")
        ax[0, j].set_title(title)
        ax[0, j].set_xlim(0, t[-1] if len(t) else 1)
        ax[0, j].set_ylim(-1, 1)
        ax[1, j].imshow(spec, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="magma",
                        extent=(0, len(x) / sr, 0, spec.shape[0]))
        ax[1, j].set_xlabel("time (s)")
    ax[0, 0].set_ylabel("amplitude")
    ax[1, 0].set_ylabel("mel bin")
    return fig


def spectrogram_png(wav: torch.Tensor, vmin: float, vmax: float, sr: int = SR,
                    width_px: int = 640, height_px: int = 200) -> bytes:
    """Log-mel spectrogram as a bare PNG (no axes) on a fixed colour scale, for the web page."""
    import io
    spec = log_mel(wav, sr)
    fig = plt.figure(figsize=(width_px / 100, height_px / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(spec, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="magma")
    ax.set_axis_off()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def shared_scale(*wavs: torch.Tensor, sr: int = SR) -> tuple[float, float]:
    specs = [log_mel(w, sr) for w in wavs]
    return float(min(s.min() for s in specs)), float(max(s.max() for s in specs))


def tradeoff_figure(points: dict, anchors: dict | None = None) -> plt.Figure:
    """Fig. 4 of the paper: EER on libri-dev-f against average UAR, one point per admixture probability.

    `points`: p -> (EER, UAR); `anchors`: label -> (EER, UAR) for the pure systems (rows 1 and 4).
    """
    fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)
    ps = sorted(points)
    ax.plot([points[p][1] for p in ps], [points[p][0] for p in ps], "o-", color="C0", label="admixture")
    for p in ps:
        ax.annotate(f"{p:g}", (points[p][1], points[p][0]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    for i, (label, (eer, uar)) in enumerate((anchors or {}).items()):
        ax.plot(uar, eer, "s", color=f"C{i + 1}", label=label)
    ax.set_xlabel("UAR (%)")
    ax.set_ylabel("EER, libri-dev-f (%)")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig
