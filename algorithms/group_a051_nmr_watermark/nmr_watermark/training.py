"""Epoch-boundary resumable training; validation-only model selection."""
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from .data import AudioDataset, manifest_hash
from .model import WatermarkModel, SpectralTransform
from .psychoacoustics import NMRLoss
from .live_progress import EpochProgress
from .prefetch import prefetch_batches


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def psycho_hash():
    return hashlib.sha256(Path(__file__).with_name("psychoacoustics.py").read_bytes()).hexdigest()


def cache_masks(manifest, output, device="cpu", batch_size=16, normalized=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = {"manifest": manifest_hash(manifest), "psycho": psycho_hash(), "normalized": normalized}
    meta_path = output / "complete.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) == identity:
        return
    seed_all(2026)
    dataset = AudioDataset(manifest)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    stft, nmr = SpectralTransform(normalized).to(device), NMRLoss(normalized).to(device)
    cache = np.lib.format.open_memmap(output / "masks.npy", mode="w+", dtype="float32", shape=(len(dataset), 109, 96))
    start = time.monotonic()
    for step, batch in enumerate(loader):
        with torch.no_grad():
            mask = nmr.masking(stft(batch["audio"].to(device)))
        if not torch.isfinite(mask).all() or (mask <= 0).any():
            raise ValueError("Invalid psychoacoustic mask")
        cache[batch["index"].numpy()] = mask.cpu().numpy()
        if step % 100 == 0:
            cache.flush()
            print(f"mask {step + 1}/{len(loader)} elapsed={time.monotonic()-start:.1f}s", flush=True)
    cache.flush()
    del cache
    meta_path.write_text(json.dumps(identity))


class TrackSampler(Sampler):
    """Shuffle 16-track pools, preserving manageable decode cache size."""
    def __init__(self, dataset, seed):
        self.rows, self.seed = dataset.rows, seed

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        groups = {}
        for i, row in enumerate(self.rows):
            groups.setdefault(row["track_id"], []).append(i)
        tracks = list(groups)
        rng.shuffle(tracks)
        for start in range(0, len(tracks), 16):
            indices = [i for t in tracks[start:start+16] for i in groups[t]]
            rng.shuffle(indices)
            yield from indices


def metrics(audio, marked_audio, logits, message, host, marked, nmr, mask):
    return {
        "ber": ((logits >= 0) != message.bool()).float().mean(1),
        "snr": 10 * torch.log10(audio.square().sum(1).clamp_min(1e-20) /
                                (audio-marked_audio).square().sum(1).clamp_min(1e-20)),
        "nmr": nmr(marked, host, mask),
    }


@torch.no_grad()
def validate(model, dataset, nmr, kind, alpha, device, batch_size, progress=None):
    model.eval()
    sums = {k: 0. for k in ["loss", "bce", "distortion", "ber", "snr", "nmr"]}
    generator = torch.Generator(device="cpu").manual_seed(19052024)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    if progress:
        progress.update('validation', 0, len(loader), force=True)
    for step, batch in enumerate(prefetch_batches(loader), 1):
        audio, mask = batch["audio"].to(device), batch["mask"].to(device)
        message = torch.randint(0, 2, (len(audio), model.bits), generator=generator).float().to(device)
        marked_audio, logits, host, marked = model(audio, message)
        values = metrics(audio, marked_audio, logits, message, host, marked, nmr, mask)
        distortion = values["nmr"] if kind == "nmr" else (marked-host).abs().square().mean((1, 2))
        bce = F.binary_cross_entropy_with_logits(logits, message, reduction="none").mean(1)
        values.update(loss=alpha*distortion+(1-alpha)*bce, bce=bce, distortion=distortion)
        for key, val in values.items():
            sums[key] += val.sum().item()
        if progress:
            progress.update('validation', step, len(loader))
    result = {key: val / len(dataset) for key, val in sums.items()}
    result["nmr_db"] = 10 * np.log10(max(result["nmr"], 1e-30))
    return result


def train(config, train_manifest, val_manifest, cache_root, output, device="cuda", resume=True, init=None):
    torch.set_num_threads(4)
    output, cache_root = Path(output), Path(cache_root)
    output.mkdir(parents=True, exist_ok=True)
    for name, manifest in [("train", train_manifest), ("validation", val_manifest)]:
        expected = {"manifest": manifest_hash(manifest), "psycho": psycho_hash(), "normalized": config.get("stft_normalized", False)}
        meta = cache_root/name/"complete.json"
        if not meta.exists() or json.loads(meta.read_text()) != expected:
            raise RuntimeError("Missing or stale mask cache: " + str(meta))
    config = dict(config, train_hash=manifest_hash(train_manifest), val_hash=manifest_hash(val_manifest), psycho_hash=psycho_hash())
    seed_all(config["seed"])
    train_data = AudioDataset(train_manifest, cache_root/"train")
    val_data = AudioDataset(val_manifest, cache_root/"validation")
    model = WatermarkModel(config["bits"], config.get("stft_normalized", False)).to(device)
    nmr = NMRLoss(config.get("stft_normalized", False)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-4), betas=(0.9, 0.999), eps=1e-8)
    start_epoch, best, stale = 0, float("inf"), 0
    last = output/"last.pt"
    source = last if resume and last.exists() else Path(init) if init else None
    if source:
        ckpt = torch.load(source, map_location=device, weights_only=False)
        if source == last and ckpt["config"] != config:
            raise ValueError("Resume configuration differs from checkpoint")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        if source == last:
            best, stale = ckpt["best"], ckpt["stale"]
        torch.set_rng_state(ckpt["rng_cpu"].cpu())
        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all([r.cpu() for r in ckpt["rng_cuda"]])
        if source == last and ckpt.get("complete"):
            return json.loads((output/"result.json").read_text())
    (output/"config.json").write_text(json.dumps(config, indent=2))
    history_path = output/"history.json"
    history = json.loads(history_path.read_text()) if source == last and history_path.exists() else []
    # Drop records from an epoch whose checkpoint never finished atomically.
    history = [h for h in history if h["epoch"] < start_epoch]
    warmup = [0.5] if config["loss"] == "mse" else [1e-4, 1e-2, 1e-1]
    for epoch in range(start_epoch, config.get("epochs", 30)):
        started = time.monotonic()
        model.train()
        alpha = warmup[epoch] if epoch < len(warmup) else config["alpha"]
        sampler = TrackSampler(train_data, config["seed"] + epoch)
        loader = DataLoader(train_data, batch_size=config["batch_size"], sampler=sampler, num_workers=0)
        progress = EpochProgress(output, epoch, config.get('epochs', 30))
        progress.update('training', 0, len(loader), force=True)
        loss_sum = 0.
        for step, batch in enumerate(prefetch_batches(loader), 1):
            audio, mask = batch["audio"].to(device), batch["mask"].to(device)
            message = torch.randint(0, 2, (len(audio), model.bits), device=device).float()
            optimizer.zero_grad(set_to_none=True)
            marked_audio, logits, host, marked = model(audio, message)
            distortion = nmr(marked, host, mask).mean() if config["loss"] == "nmr" else (marked-host).abs().square().mean()
            loss = alpha*distortion + (1-alpha)*F.binary_cross_entropy_with_logits(logits, message)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss in epoch {epoch}")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()*len(audio)
            progress.update('training', step, len(loader))
        scores = validate(model, val_data, nmr, config["loss"], alpha, device, config["batch_size"], progress)
        progress.update('saving', 0, 1, force=True)
        # Warmup losses at different alphas are not comparable to final-stage loss.
        eligible = epoch >= len(warmup) or config.get("warmup_only", False)
        improved = eligible and scores["loss"] < best
        if eligible:
            if improved:
                best, stale = scores["loss"], 0
            else:
                stale += 1
        complete = epoch+1 == config.get("epochs", 30) or (eligible and stale >= 2)
        row = {"epoch": epoch, "alpha": alpha, "train_loss": loss_sum/len(train_data),
               "seconds": time.monotonic()-started, **scores}
        history.append(row)
        history_path.write_text(json.dumps(history, indent=2))
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                 "config": config, "best": best, "stale": stale, "metrics": scores,
                 "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                 "complete": complete}
        if improved:
            atomic_save({"model": model.state_dict(), "config": config, "epoch": epoch, "metrics": scores}, output/"best.pt")
        if complete:
            best_file = output/"best.pt"
            result = torch.load(best_file, map_location="cpu", weights_only=False)["metrics"] if best_file.exists() else scores
            (output/"result.json").write_text(json.dumps({"config": config, "metrics": result, "epochs": epoch+1}, indent=2))
        atomic_save(state, last)
        progress.update('complete' if complete else 'epoch_complete', 1, 1, force=True)
        print(json.dumps(row), flush=True)
        if complete:
            break
    return json.loads((output/"result.json").read_text())


def candidates(batch_size=16):
    for bits in (256, 512):
        for loss, alphas in [("mse", [.8, .9, .92, .95, .98]), ("nmr", [.1, .2, .3, .4, .5])]:
            for seed in (2026, 2027, 2028):
                for alpha in alphas:
                    yield {"bits": bits, "loss": loss, "seed": seed, "alpha": alpha, "batch_size": batch_size, "lr": 1e-4, "epochs": 30, "stft_normalized": True}


def search(manifest_dir, cache_root, output, device="cuda", batch_size=16):
    manifest_dir, output = Path(manifest_dir), Path(output)
    summary = json.loads((manifest_dir/"summary.json").read_text())
    if summary["pilot"] or summary["hours"]["train"] < 80 or summary["hours"]["validation"] < 10:
        raise ValueError("Full 60-model search requires full verified 80h/10h data")
    jobs = list(candidates(batch_size))
    output.mkdir(parents=True, exist_ok=True)
    (output/"queue.json").write_text(json.dumps(jobs, indent=2))
    for config in jobs:
        tag = f'{config["loss"]}{config["bits"]}_s{config["seed"]}'
        warmdir = output/"warmup"/tag
        warmconfig = dict(config, warmup_only=True, epochs=1 if config["loss"] == "mse" else 3, alpha=0.)
        train(warmconfig, manifest_dir/"train.json", manifest_dir/"validation.json", cache_root, warmdir, device)
        train(config, manifest_dir/"train.json", manifest_dir/"validation.json", cache_root,
              output/f'{tag}_a{config["alpha"]}', device, init=warmdir/"last.pt")


def select(output):
    output = Path(output)
    results = [(p.parent, json.loads(p.read_text())) for p in output.glob("*/result.json")]
    if len(results) != 60:
        raise ValueError(f"Expected 60 completed candidate runs; found {len(results)}")
    chosen = {}
    for bits in (256, 512):
        subset = [(p, r) for p, r in results if r["config"]["bits"] == bits]
        mse = max([(p, r) for p, r in subset if r["config"]["loss"] == "mse"], key=lambda pr: pr[1]["metrics"]["snr"])
        valid = [(p, r) for p, r in subset if r["config"]["loss"] == "nmr" and r["metrics"]["ber"] < mse[1]["metrics"]["ber"]]
        if not valid:
            raise RuntimeError(f"No NMR{bits} candidate meets paper BER constraint")
        nmr = min(valid, key=lambda pr: pr[1]["metrics"]["nmr"])
        for name, (path, result) in [(f"MSE{bits}", mse), (f"NMR{bits}", nmr)]:
            chosen[name] = {"checkpoint": str((path/"best.pt").resolve()), "metrics": result["metrics"]}
    (output/"selected.json").write_text(json.dumps(chosen, indent=2))
    return chosen
