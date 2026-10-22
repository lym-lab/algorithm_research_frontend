"""Train and apply the WavLM conversion model (Table 1 rows 5-8).

    # 1. fixed target speaker a and its k-means units
    python -m a082.wavlm_conv.train prepare-target --manifest pool.json --pool <cache> --out work/wc/target.pt [--speaker idXXXXX]
    # 2. speaker classifier for the adversarial loss, pretrained on clean features
    python -m a082.wavlm_conv.train pretrain-spk --vpc-root R --out work/wc/spk.pt
    # 3. base model (row 5), then the variants from the base checkpoint
    python -m a082.wavlm_conv.train train --config configs/wc5.yaml --vpc-root R --target work/wc/target.pt --manifest pool.json
    python -m a082.wavlm_conv.train train --config configs/wc6.yaml ... --spk work/wc/spk.pt --init work/wc5/last.pt
    # the same on n GPUs of one machine (data parallel; `batch` in the config stays the total batch)
    OMP_NUM_THREADS=8 torchrun --standalone --nproc_per_node 4 -m a082.wavlm_conv.train train --config ...
    # 4. anonymize the VPC sets (same runner as the other systems)
    python -m a082.wavlm_conv.train infer --ckpt work/wc5/last.pt --suffix _wc5 --vpc-root R [--sets all --shard i/n]

Training data: LibriSpeech train-clean-360 through the VPC data directory. Inputs w(u) and targets
k(w(u), a) are computed on the GPU at every step and never stored (one copy would be ~133 GB).
"""
from __future__ import annotations

import argparse
import json
import sys
import math
import os
import random
import signal
import time
from pathlib import Path

import soundfile as sf
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from .. import audio, kaldi, runner, seeding
from ..knnvc.core import knn_regress
from ..knnvc.pool import Pool
from ..models import load_hifigan, load_wavlm, vocode, FEATURE_LAYER
from . import kmeans, losses
from .model import NoamLR, WavLMConverter, count_params, lengths_to_pad


# ---------------------------------------------------------------- data

class UtteranceSet(Dataset):
    """(wav[S], speaker index) from a Kaldi directory; long utterances are randomly cropped."""

    def __init__(self, vpc_root: Path, name: str, utts: list[str], spk_index: dict[str, int], crop_s: float):
        d = vpc_root / "data" / name
        self.root, self.scp, self.u2s = vpc_root, kaldi.read_kaldi(d / "wav.scp"), kaldi.read_kaldi(d / "utt2spk")
        self.utts, self.spk_index, self.crop = utts, spk_index, int(crop_s * audio.SR)

    def __len__(self):
        return len(self.utts)

    def __getitem__(self, i):
        u = self.utts[i]
        wav, sr = audio.load_scp_entry(self.scp[u], root=self.root)
        wav = audio.resample(wav, sr)[0]
        if wav.shape[0] > self.crop:
            start = random.randrange(wav.shape[0] - self.crop)
            wav = wav[start:start + self.crop]
        return wav, self.spk_index[self.u2s[u]]


class FileSet(Dataset):
    """Target-speaker utterances for the target-reconstruction samples; `crop_s=None` keeps them whole."""

    def __init__(self, files: list[str], crop_s: float | None):
        self.files, self.crop = files, int(crop_s * audio.SR) if crop_s else None
        self._trimmed: dict[int, torch.Tensor] = {}  # VAD is slow on CPU: trim each file once per worker

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        if i not in self._trimmed:
            self._trimmed[i] = audio.vad_trim(audio.load_audio(self.files[i]))[0]
        wav = self._trimmed[i]
        if self.crop and wav.shape[0] > self.crop:
            start = random.randrange(wav.shape[0] - self.crop)
            wav = wav[start:start + self.crop]
        return wav, -1


def collate(batch):
    wavs, spk = zip(*batch)
    n = torch.tensor([w.shape[0] for w in wavs])
    x = torch.zeros(len(wavs), int(n.max()))
    for i, w in enumerate(wavs):
        x[i, : w.shape[0]] = w
    return x, n, torch.tensor(spk)


def forever(loader):
    epoch = 0
    while True:
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)  # a new shuffle every epoch, the same on all ranks
        yield from loader
        epoch += 1


def split_train_valid(utts: list[str], valid_frac: float = 0.005) -> tuple[list[str], list[str]]:
    valid = [u for u in utts if seeding.uniform01(u, "wc-valid") < valid_frac]
    vs = set(valid)
    return [u for u in utts if u not in vs], valid


def speaker_index(vpc_root: Path, name: str = kaldi.TRAIN_SET) -> dict[str, int]:
    spks = sorted(set(kaldi.read_kaldi(vpc_root / "data" / name / "utt2spk").values()))
    return {s: i for i, s in enumerate(spks)}


# ---------------------------------------------------------------- data parallel (torchrun)
#
# One process per GPU. Each rank draws its share of the batch, gradients are averaged over the ranks before
# every optimizer step, and batch norm is synchronised, so one step still sees `batch` utterances and the step
# counts, learning-rate schedule and checkpoints are those of one GPU. The losses are means within each rank,
# so the averaged gradient weights the ranks equally rather than by their number of frames.

def init_dist(args) -> tuple[int, int]:
    """(rank, world size); (0, 1) when not started by torchrun. Sets args.device to this rank's GPU."""
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world == 1:
        return 0, 1
    cuda = str(args.device).startswith("cuda")
    if cuda:
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        args.device = f"cuda:{local}"
    dist.init_process_group("nccl" if cuda else "gloo")
    return dist.get_rank(), world


def rank_share(total: int, rank: int, world: int) -> int:
    """This rank's part of `total` items; the remainder goes to the lowest ranks, so the parts sum to `total`."""
    return total // world + (rank < total % world)


def all_reduce_grads(params, world: int) -> None:
    """Average the gradients over the ranks, in one flat buffer (a missing gradient counts as zero)."""
    if world == 1:
        return
    ps = [p for p in params if p.requires_grad]
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in ps]
    flat = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat)
    flat /= world
    for p, g in zip(ps, torch._utils._unflatten_dense_tensors(flat, grads)):
        if p.grad is None:
            p.grad = g
        else:
            p.grad.copy_(g)


def data_parallel(module: nn.Module, world: int, dev) -> nn.Module:
    """Synchronised batch norm (works without DDP in torch 2.0) and rank 0's weights on every rank."""
    if world == 1:
        return module
    if str(dev).startswith("cuda"):
        module = nn.SyncBatchNorm.convert_sync_batchnorm(module)
    for t in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(t.data, 0)
    return module


# ---------------------------------------------------------------- features on the GPU

def target_features(files: list[str], wavlm, dev, cache: Path, workers: int, amp: bool) -> list[torch.Tensor]:
    """Layer-6 WavLM features of the VAD-trimmed target-speaker files, computed once and cached (fp16) next to
    target.pt. WavLM is frozen, so recomputing them for the 3 target samples of every step only cost time
    (about 8 % of a step, 2026-10-22); a 10 s training crop is then taken in frames."""
    if cache.exists():
        feats = torch.load(cache, map_location="cpu")
        if len(feats) == len(files):
            return [f.to(dev) for f in feats]
    from ..models import wavlm_features
    feats = []
    loader = DataLoader(FileSet(files, None), batch_size=None, num_workers=workers)
    with torch.inference_mode():
        for wav, _ in loader:
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                feats.append(wavlm_features(wavlm, wav[None].to(dev)).half().cpu())
    tmp = cache.with_suffix(".tmp")
    torch.save(feats, tmp); tmp.replace(cache)
    return [f.to(dev) for f in feats]


def target_batch(feats: list[torch.Tensor], n: int, crop_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    """n random target utterances, each cropped to at most crop_frames at a random start: [n, T, D], lengths."""
    crops = []
    for _ in range(n):
        f = feats[random.randrange(len(feats))]
        if f.shape[0] > crop_frames:
            start = random.randrange(f.shape[0] - crop_frames)
            f = f[start:start + crop_frames]
        crops.append(f)
    lengths = torch.tensor([f.shape[0] for f in crops], device=crops[0].device)
    x = torch.zeros(n, int(lengths.max()), crops[0].shape[1], device=crops[0].device)
    for i, f in enumerate(crops):
        x[i, : f.shape[0]] = f.float()
    return x, lengths


def frames_for(samples: torch.Tensor) -> torch.Tensor:
    """Output length of WavLM's convolutional front end (kernels 10,3,3,3,3,2,2; strides 5,2,2,2,2,2,2)."""
    n = samples.clone()
    for k, s in [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]:
        n = torch.div(n - k, s, rounding_mode="floor") + 1
    return n


@torch.no_grad()
def batch_features(wavlm, wav: torch.Tensor, n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Padded batch wav[B, S] -> layer-6 features [B, T, 1024] and frame lengths [B].
    Padding is masked in WavLM's self-attention; padded frames are zeroed."""
    pad = torch.arange(wav.shape[1], device=wav.device)[None] >= n[:, None].to(wav.device)
    x = wavlm.extract_features(wav, padding_mask=pad, output_layer=FEATURE_LAYER, ret_layer_results=False)[0]
    lengths = frames_for(n).clamp(max=x.shape[1]).to(x.device)
    x = x.masked_fill(lengths_to_pad(lengths, x.shape[1])[..., None], 0.0)
    return x, lengths


@torch.no_grad()
def knn_targets(x: torch.Tensor, lengths: torch.Tensor, matching: torch.Tensor) -> torch.Tensor:
    y = torch.zeros_like(x)
    for b in range(x.shape[0]):
        t = int(lengths[b])
        y[b, :t] = knn_regress(x[b, :t], matching, k=4)
    return y


# ---------------------------------------------------------------- speaker classifier

class SpeakerClassifier(nn.Module):
    """ECAPA-TDNN on 1024-dim WavLM features (SpeechBrain network definition, trained by us)."""

    def __init__(self, n_speakers: int, feat_dim: int = 1024, emb: int = 192):
        super().__init__()
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
        self.ecapa = ECAPA_TDNN(feat_dim, lin_neurons=emb)
        self.head = nn.Linear(emb, n_speakers)
        self.n_speakers = n_speakers

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        rel = lengths.float() / x.shape[1]
        return self.head(self.ecapa(x, rel).squeeze(1))


# ---------------------------------------------------------------- commands

def cmd_prepare_target(args) -> None:
    manifest = json.loads(Path(args.manifest).read_text())
    pool = Pool(args.pool, device="cpu", min_frames=1)
    spk = args.speaker
    if spk is None:  # the pool speaker with the most audio (raw duration), at least 20 min
        dur = {s: sum(sf.info(f).duration for f in manifest[s]) for s in pool.speakers if s in manifest}
        spk = max(dur, key=dur.get)
        if dur[spk] < 20 * 60:
            raise RuntimeError(f"longest speaker {spk} has only {dur[spk] / 60:.1f} min")
    matching = pool.get(spk)
    centroids = kmeans.spherical_kmeans(matching.to(args.device), k=args.k).cpu()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"speaker": spk, "matching": matching.half(), "centroids": centroids, "k": args.k}, args.out)
    print(f"target {spk}: {matching.shape[0]} frames, {args.k} centroids -> {args.out}")


def cmd_pretrain_spk(args) -> None:
    dev = args.device
    spk_index = speaker_index(args.vpc_root)
    utts = sorted(kaldi.read_kaldi(args.vpc_root / "data" / kaldi.TRAIN_SET / "wav.scp"))
    train_u, valid_u = split_train_valid(utts)
    mk = lambda us, shuffle: DataLoader(UtteranceSet(args.vpc_root, kaldi.TRAIN_SET, us, spk_index, args.crop),
                                        batch_size=args.batch, shuffle=shuffle, num_workers=args.workers,
                                        collate_fn=collate, drop_last=shuffle)
    train_dl, valid_dl = mk(train_u, True), mk(valid_u, False)
    wavlm = load_wavlm(dev)
    feat_dim = wavlm.cfg.encoder_embed_dim
    clf = SpeakerClassifier(len(spk_index), feat_dim).to(dev)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    step, it = 0, forever(train_dl)
    while step < args.steps:
        wav, n, spk = next(it)
        x, lengths = batch_features(wavlm, wav.to(dev), n)
        loss = F.cross_entropy(clf(x, lengths), spk.to(dev))
        opt.zero_grad(); loss.backward(); opt.step(); step += 1
        if step % args.log_every == 0:
            print(f"step {step} ce {loss.item():.3f}", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            acc = accuracy(clf, wavlm, valid_dl, dev)
            print(f"step {step} valid acc {acc:.4f}", flush=True)
            torch.save({"model": clf.state_dict(), "n_speakers": len(spk_index), "feat_dim": feat_dim,
                        "valid_acc": acc}, args.out)
            if acc >= args.target_acc:
                break


@torch.no_grad()
def accuracy(clf, wavlm, loader, dev, convert=None) -> float:
    clf.eval()
    right = total = 0
    for wav, n, spk in loader:
        x, lengths = batch_features(wavlm, wav.to(dev), n)
        if convert is not None:
            x = convert(x, lengths)
        right += (clf(x, lengths).argmax(-1).cpu() == spk).sum().item()
        total += len(spk)
    clf.train()
    return right / max(total, 1)


def cmd_train(args) -> None:
    cfg = yaml.safe_load(Path(args.config).read_text())
    rank, world = init_dist(args)
    dev = args.device
    say = print if rank == 0 else (lambda *a, **k: None)
    if cfg["batch"] % world:
        raise ValueError(f"batch {cfg['batch']} is not divisible by {world} GPUs")
    batch = cfg["batch"] // world
    out = Path(args.out or f"work/{cfg['name']}")
    out.mkdir(parents=True, exist_ok=True)
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed); random.seed(seed)

    target = torch.load(args.target, map_location="cpu")
    matching = target["matching"].float().to(dev)
    spk_index = speaker_index(args.vpc_root)
    utts = sorted(kaldi.read_kaldi(args.vpc_root / "data" / kaldi.TRAIN_SET / "wav.scp"))
    train_u, valid_u = split_train_valid(utts)
    src_set = UtteranceSet(args.vpc_root, kaldi.TRAIN_SET, train_u, spk_index, cfg["crop_s"])
    sampler = DistributedSampler(src_set, world, rank, shuffle=True, seed=seed, drop_last=True) if world > 1 else None
    src_dl = DataLoader(src_set, batch_size=batch, shuffle=sampler is None, sampler=sampler, num_workers=args.workers,
                        collate_fn=collate, drop_last=True, persistent_workers=args.workers > 0)
    valid_dl = DataLoader(UtteranceSet(args.vpc_root, kaldi.TRAIN_SET, valid_u[: cfg.get("valid_utts", 200)], spk_index,
                                       cfg["crop_s"]), batch_size=batch, num_workers=args.workers, collate_fn=collate)
    rho = cfg.get("target_recon", 0.2)
    n_tgt = rank_share(int(round(rho * cfg["batch"])), rank, world)  # 3 of 16: 1, 1, 1, 0 on 4 GPUs
    src_it = forever(src_dl)

    wavlm = load_wavlm(dev)
    model = data_parallel(WavLMConverter(**cfg.get("model", {})).to(dev), world, dev)
    ctc = losses.CosineCTC(target["centroids"], sigma=cfg.get("ctc_sigma", 10.0)).to(dev) if cfg.get("ctc") else None
    params = list(model.parameters()) + (list(ctc.parameters()) if ctc is not None else [])
    opt = torch.optim.Adam(params, lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    sched = NoamLR(opt, d=model.d, warmup=cfg.get("warmup", 4000))
    clf = clf_opt = None
    if cfg.get("adv"):
        ck = torch.load(args.spk, map_location="cpu")
        clf = SpeakerClassifier(ck["n_speakers"], ck.get("feat_dim", 1024)).to(dev)
        clf.load_state_dict(ck["model"])
        clf = data_parallel(clf, world, dev)
        clf_opt = torch.optim.Adam(clf.parameters(), lr=cfg.get("clf_lr", 1e-4))
    # mixed precision (--amp, CUDA only): WavLM, the kNN similarities and the converter run in fp16 autocast, the
    # losses in fp32, with loss scaling; 2.4x faster per step on a V100 (2026-10-22), where fp32 put the
    # voucher budget for ID 5-8 out of reach. Steps, batch and schedule are unchanged.
    amp = bool(getattr(args, "amp", False)) and str(dev).startswith("cuda")
    autocast = lambda: torch.autocast("cuda", dtype=torch.float16, enabled=amp)
    tfeats = None
    if n_tgt:  # (until 2026-10-22 the audio was loaded, VAD-trimmed and run through WavLM at every step)
        files = json.loads(Path(args.manifest).read_text())[target["speaker"]]
        tfeats = target_features(files, wavlm, dev, Path(args.target).with_name("target_feats.pt"), args.workers, amp)
        crop_frames = int(frames_for(torch.tensor([int(cfg["crop_s"] * audio.SR)]))[0])
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    clf_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    step = 0
    if args.init:  # variants continue from the base checkpoint; optimizer and schedule restart
        model.load_state_dict(torch.load(args.init, map_location="cpu")["model"])
    resume = out / "last.pt"
    if resume.exists():
        ck = torch.load(resume, map_location="cpu")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        if ctc is not None and "ctc" in ck: ctc.load_state_dict(ck["ctc"])
        if clf is not None and "clf" in ck: clf.load_state_dict(ck["clf"]); clf_opt.load_state_dict(ck["clf_opt"])
        if amp and "scaler" in ck: scaler.load_state_dict(ck["scaler"])
        step = ck["step"]
    if world > 1:  # the same weights on all ranks, then a different random stream (dropout, crops) on each
        torch.manual_seed(seed + rank); random.seed(seed + rank)
    say(f"{cfg['name']}: {count_params(model):,} parameters, starting at step {step}, {world} GPU(s) x batch {batch}"
        + (", fp16 autocast" if amp else ""), flush=True)

    def save() -> None:
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
              "cfg": cfg, "target_speaker": target["speaker"], "gpus": world}
        if ctc is not None: ck["ctc"] = ctc.state_dict()
        if clf is not None: ck.update(clf=clf.state_dict(), clf_opt=clf_opt.state_dict())
        if amp: ck["scaler"] = scaler.state_dict()
        torch.save(ck, out / "last.tmp"); (out / "last.tmp").replace(resume)

    # one process: SIGTERM (queue restart, guard, shutdown) saves the current step before exiting; under torchrun
    # the ranks could stop at different steps, so there the last periodic checkpoint is the restart point
    stop = []
    if world == 1:
        signal.signal(signal.SIGTERM, lambda *_: stop.append(1))

    lam = cfg.get("adv_lambda", 0.1)
    mu = cfg.get("ctc_weight", 1.0)
    t0 = time.time()
    while step < cfg["steps"]:
        wav, n, spk = next(src_it)
        with autocast():
            x, lengths = batch_features(wavlm, wav.to(dev), n)
            y = knn_targets(x, lengths, matching)
        x = x.float()
        spk = spk.to(dev)
        is_src = torch.ones(x.shape[0], dtype=torch.bool, device=dev)
        if tfeats is not None:  # replace the last n_tgt samples by (w(u_a), w(u_a)) pairs
            tx, tlen = target_batch(tfeats, n_tgt, crop_frames)
            T = max(x.shape[1], tx.shape[1])
            x, y, tx = [F.pad(a, (0, 0, 0, T - a.shape[1])) for a in (x, y, tx)]
            x = torch.cat([x[: -n_tgt], tx]); y = torch.cat([y[: -n_tgt], tx])
            lengths = torch.cat([lengths[: -n_tgt], tlen]); spk = spk[: x.shape[0]]
            is_src[-n_tgt:] = False
        pad = lengths_to_pad(lengths, x.shape[1])
        with autocast():
            y_hat, y_post = model(x, pad)
        y_hat, y_post = y_hat.float(), y_post.float()
        loss = losses.reconstruction(y_hat, y_post, y, ~pad)
        log = {"rec": loss.item()}

        if clf is not None:
            # classifier step on detached outputs of source utterances only
            src = is_src.nonzero().squeeze(1)
            with autocast():
                c_logits = clf(y_post[src].detach(), lengths[src])
            c_loss = F.cross_entropy(c_logits.float(), spk[src])
            clf_opt.zero_grad(); clf_scaler.scale(c_loss).backward(); all_reduce_grads(clf.parameters(), world)
            clf_scaler.step(clf_opt); clf_scaler.update()
            with autocast():
                a_logits = clf(y_post[src], lengths[src])
            adv = losses.capped_adversarial(a_logits.float(), spk[src], clf.n_speakers)
            loss = loss - lam * adv
            log.update(clf_ce=c_loss.item(), adv=adv.item())
        if ctc is not None:
            units = [losses.dedup(kmeans.assign(y[b, : lengths[b]], ctc.centroids)) for b in range(y.shape[0])]
            c = ctc(y_post, lengths, units)
            loss = loss + mu * c
            log["ctc"] = c.item()

        opt.zero_grad(); scaler.scale(loss).backward(); all_reduce_grads(params, world)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, cfg.get("grad_clip", 1.0))
        scaler.step(opt); scaler.update(); sched.step(); step += 1

        if step % args.log_every == 0:
            rate = args.log_every / (time.time() - t0); t0 = time.time()
            say(f"step {step} " + " ".join(f"{k} {v:.4f}" for k, v in log.items()) + f" | {rate:.2f} it/s", flush=True)
        if step % args.save_every == 0 or step == cfg["steps"]:
            if rank == 0:
                save()
                with autocast():
                    v = validate(model, wavlm, valid_dl, matching, dev)
                print(f"step {step} valid rec {v:.4f}", flush=True)
            if world > 1:
                dist.barrier()  # the other ranks wait while rank 0 saves and validates
        if stop:
            save()
            print(f"SIGTERM: saved step {step}, exiting", flush=True)
            sys.exit(143)
    if world > 1:
        dist.destroy_process_group()


@torch.no_grad()
def validate(model, wavlm, loader, matching, dev) -> float:
    model.eval()
    total = n_b = 0
    for wav, n, _ in loader:
        x, lengths = batch_features(wavlm, wav.to(dev), n)
        y = knn_targets(x, lengths, matching)
        pad = lengths_to_pad(lengths, x.shape[1])
        y_hat, y_post = model(x, pad)
        total += losses.reconstruction(y_hat.float(), y_post.float(), y.float(), ~pad).item(); n_b += 1
    model.train()
    return total / max(n_b, 1)


def cmd_infer(args) -> None:
    dev = args.device
    model = wavlm = hifigan = None
    if not args.finalize_only:
        ck = torch.load(args.ckpt, map_location="cpu")
        model = WavLMConverter(**ck["cfg"].get("model", {})).to(dev).eval()
        model.load_state_dict(ck["model"])
        wavlm, hifigan = load_wavlm(dev), load_hifigan(dev, prematched=True)

    # --amp: WavLM, converter and HiFi-GAN under fp16 autocast (the vocoder 2.6x faster on a V100, 2026-10-22);
    # the loudness is measured on the fp32 waveform
    amp = bool(getattr(args, "amp", False)) and str(dev).startswith("cuda")

    @torch.inference_mode()
    def convert(utt, wav):
        from ..models import wavlm_features
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            x = wavlm_features(wavlm, wav).float()[None]
            _, y = model(x)
            out = vocode(hifigan, y[0].float())
        return audio.loudness_normalize(out.float())

    runner.anonymize_sets(args.vpc_root, runner.resolve_sets(args.sets), args.suffix,
                          None if args.finalize_only else convert,
                          shard=runner.parse_shard(args.shard), finalize_only=args.finalize_only)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare-target")
    a.add_argument("--manifest", required=True); a.add_argument("--pool", required=True); a.add_argument("--out", required=True)
    a.add_argument("--speaker"); a.add_argument("--k", type=int, default=100); a.add_argument("--device", default="cuda")
    a = sub.add_parser("pretrain-spk")
    a.add_argument("--vpc-root", required=True, type=Path); a.add_argument("--out", required=True)
    a.add_argument("--steps", type=int, default=30000); a.add_argument("--batch", type=int, default=32)
    a.add_argument("--crop", type=float, default=4.0); a.add_argument("--target-acc", type=float, default=0.99)
    a.add_argument("--eval-every", type=int, default=2000); a.add_argument("--log-every", type=int, default=100)
    a.add_argument("--workers", type=int, default=8); a.add_argument("--device", default="cuda")
    a = sub.add_parser("train")
    a.add_argument("--config", required=True); a.add_argument("--vpc-root", required=True, type=Path)
    a.add_argument("--target", required=True); a.add_argument("--manifest", required=True)
    a.add_argument("--spk"); a.add_argument("--init"); a.add_argument("--out")
    a.add_argument("--workers", type=int, default=8); a.add_argument("--device", default="cuda")
    a.add_argument("--log-every", type=int, default=100); a.add_argument("--save-every", type=int, default=5000)
    a.add_argument("--amp", action="store_true", help="fp16 autocast with loss scaling (CUDA only)")
    a = sub.add_parser("infer")
    a.add_argument("--ckpt"); a.add_argument("--suffix", required=True)
    a.add_argument("--amp", action="store_true", help="fp16 autocast for WavLM, converter and vocoder (CUDA only)")
    runner.add_common_args(a)
    args = ap.parse_args(argv)
    {"prepare-target": cmd_prepare_target, "pretrain-spk": cmd_pretrain_spk,
     "train": cmd_train, "infer": cmd_infer}[args.cmd](args)


if __name__ == "__main__":
    main()
