import csv
import json
import math
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import DataLoader
from .data import AudioDataset, manifest_hash
from .inference import load_model
from .psychoacoustics import NMRLoss
from .training import metrics


def evaluate(checkpoint, manifest, output, device="cpu", peaq_audio=False):
    model, config = load_model(checkpoint, device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = AudioDataset(manifest)
    nmr = NMRLoss(config.get("stft_normalized", False)).to(device)
    # Same full 512-bit message stream for both payload lengths; 256 uses prefix.
    rng = np.random.default_rng(240815553)
    messages = rng.integers(0, 2, (len(dataset), 512)).astype(np.float32)
    rows, peaq_rows = [], []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=8, shuffle=False):
            x = batch["audio"].to(device)
            m = torch.from_numpy(messages[batch["index"].numpy(), :model.bits]).to(device)
            y, logits, host, marked = model(x, m)
            values = metrics(x, y, logits, m, host, marked, nmr, None)
            for j, index in enumerate(batch["index"].tolist()):
                item = {"index": index, "track_id": dataset.rows[index]["track_id"],
                        **{key: float(value[j]) for key, value in values.items()}}
                item["nmr_db"] = 10*np.log10(max(item["nmr"], 1e-30))
                rows.append(item)
                if peaq_audio:
                    pair = []
                    for label, audio in [("reference", x[j].cpu().numpy()), ("marked", y[j].cpu().numpy())]:
                        file = output/"peaq_audio"/f"{index:04d}_{label}.wav"
                        file.parent.mkdir(exist_ok=True)
                        sf.write(file, resample_poly(audio, 160, 147), 48000, subtype="FLOAT")
                        pair.append(str(file.resolve()))
                    peaq_rows.append([index, *pair])
    with (output/"metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if peaq_audio:
        with (output/"peaq_pairs.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "reference", "marked"])
            writer.writerows(peaq_rows)
    summary = {"samples": len(rows), "manifest_sha256": manifest_hash(manifest), "config": config,
               "ber": float(np.mean([r["ber"] for r in rows])),
               "snr": float(np.mean([r["snr"] for r in rows])),
               "nmr_db": float(10*np.log10(max(np.mean([r["nmr"] for r in rows]), 1e-30))),
               "peaq": "pending_matlab" if peaq_audio else "not_run"}
    (output/"summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def figures(results_root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(results_root)
    names = ["MSE256", "NMR256", "MSE512", "NMR512"]
    metrics_files = [root/name/"metrics.csv" for name in names]
    if not all(p.exists() for p in metrics_files):
        raise FileNotFoundError("All four real model evaluations are required")
    data = [np.genfromtxt(p, delimiter=",", names=True, dtype=None, encoding="utf-8") for p in metrics_files]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.boxplot([np.atleast_1d(d["snr"]) for d in data], tick_labels=names, whis=(5, 95), showfliers=False)
    ax.set_ylabel("SNR (dB)")
    fig.tight_layout()
    fig.savefig(root/"snr.png", dpi=180)
    plt.close(fig)
    peaq = [root/name/"peaq.csv" for name in names]
    if all(p.exists() for p in peaq):
        odg = [np.genfromtxt(p, delimiter=",", names=True)["odg"] for p in peaq]
        common = np.logical_and.reduce([np.isfinite(values) for values in odg])
        if not common.any():
            raise ValueError("No common finite PEAQ samples")
        odg = [values[common] for values in odg]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.boxplot(odg, tick_labels=names, whis=(5, 95), showfliers=False)
        ax.set_ylabel(f"PEAQ ODG (common n={int(common.sum())})")
        fig.tight_layout()
        fig.savefig(root/"odg.png", dpi=180)
        plt.close(fig)
