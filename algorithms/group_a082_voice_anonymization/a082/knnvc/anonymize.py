"""Systems 1, 1M, 2 and 3 of Table 1: kNN-VC with random VoxCeleb targets and its variations.

    python -m a082.knnvc.anonymize --config configs/knn1.yaml --vpc-root <VPC repo> --pool <cache dir>
           [--sets all|eval|<name>,...] [--shard i/n] [--device cuda] [--finalize-only]

Writes `<vpc-root>/data/<set><suffix>/` for every set.
With n shards, run one process per shard (e.g. one per GPU), then once with --finalize-only.

Random choices use separate keys, shared by all kNN systems: an utterance gets the same target
speaker in systems 1, 2 and 3, so their differences come only from the added modifications.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from .. import audio, kaldi, seeding
from ..models import load_hifigan, load_wavlm, vocode, wavlm_features
from .core import knn_pooled, stretch, stretched_frames, uniform_noise
from .pool import Pool


class KnnSystem:
    """One configured kNN system. `cfg` keys (see configs/knn*.yaml):
    k, n_targets, dirichlet (bool), length_range ([lo, hi] or null),
    noise ({where: matching|query, delta: float} or null), loudness_lufs.
    """

    def __init__(self, cfg: dict, pool: Pool, wavlm, hifigan, device):
        self.cfg, self.pool, self.wavlm, self.hifigan, self.device = cfg, pool, wavlm, hifigan, device

    def plan(self, utt: str) -> dict:
        """All random choices for `utt`, without touching audio."""
        c = self.cfg
        r = seeding.rng(utt, "target")
        targets = [self.pool.speakers[i] for i in r.choice(len(self.pool), size=c["n_targets"], replace=False)]
        if c["n_targets"] > 1 and c.get("dirichlet", True):
            weights = seeding.rng(utt, "pool-weights").dirichlet(np.ones(c["n_targets"]))
        else:
            weights = np.full(c["n_targets"], 1.0 / c["n_targets"])
        ratio = None
        if c.get("length_range"):
            lo, hi = c["length_range"]
            ratio = float(seeding.rng(utt, "length").uniform(lo, hi))
        return {"targets": targets, "weights": weights, "ratio": ratio}

    @torch.inference_mode()
    def convert(self, utt: str, wav: torch.Tensor, plan: dict | None = None) -> torch.Tensor:
        """wav[1, T] at 16 kHz -> anonymized wav[1, T'] at 16 kHz."""
        c = self.cfg
        plan = plan or self.plan(utt)
        x = wavlm_features(self.wavlm, wav).float()
        if plan["ratio"] is not None:
            x = stretch(x, stretched_frames(wav.shape[-1], plan["ratio"]))
        sets = [self.pool.get(t) for t in plan["targets"]]
        noise = c.get("noise")
        if noise:
            g = seeding.torch_gen(utt, "noise", device=x.device)
            if noise["where"] == "query":
                x = x + uniform_noise(x.shape, noise["delta"], g, x.device)
            elif noise["where"] == "matching":
                sets = [s + uniform_noise(s.shape, noise["delta"], g, s.device) for s in sets]
            else:
                raise ValueError(f"noise.where = {noise['where']!r}")
        y = knn_pooled(x, sets, torch.as_tensor(plan["weights"], dtype=torch.float32), k=c["k"])
        out = vocode(self.hifigan, y)
        return audio.loudness_normalize(out, target_lufs=c.get("loudness_lufs", -16.0))


def resolve_sets(spec: str) -> list[str]:
    if spec == "all":
        return kaldi.ALL_SETS
    if spec == "eval":
        return kaldi.EVAL_SETS
    return spec.split(",")


def run(cfg: dict, vpc_root: Path, pool_dir: Path, sets: list[str], shard=(0, 1), device="cuda",
        finalize_only=False, wavlm_path=None, hifigan_path=None) -> None:
    data = vpc_root / "data"
    suffix = cfg["suffix"]
    if not finalize_only:
        pool = Pool(pool_dir, device=device)
        system = KnnSystem(cfg, pool, load_wavlm(device, wavlm_path),
                           load_hifigan(device, cfg.get("prematched", True), hifigan_path), device)
        for name in sets:
            src, out = data / name, data / f"{name}{suffix}"
            kaldi.prepare_output_dir(src, out)
            scp = kaldi.read_kaldi(src / "wav.scp")
            utts = sorted(scp)[shard[0]::shard[1]]
            plans = {u: system.plan(u) for u in utts}
            utts.sort(key=lambda u: (plans[u]["targets"], u))  # consecutive utterances share matching sets
            for u in tqdm(utts, desc=f"{name}{suffix} [{shard[0]}/{shard[1]}]"):
                dst = kaldi.wav_path(out, u)
                if dst.exists():
                    continue
                wav, sr = audio.load_scp_entry(scp[u], root=vpc_root)
                wav = audio.resample(wav, sr)
                audio.save_wav16(dst, system.convert(u, wav, plans[u]))
    if finalize_only or shard[1] == 1:
        for name in sets:
            kaldi.finalize_output_dir(data / name, data / f"{name}{suffix}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--vpc-root", required=True, type=Path)
    ap.add_argument("--pool", type=Path, help="matching-set cache built by a082.knnvc.pool")
    ap.add_argument("--sets", default="all")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--finalize-only", action="store_true")
    ap.add_argument("--wavlm"); ap.add_argument("--hifigan")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    i, n = map(int, args.shard.split("/"))
    run(cfg, args.vpc_root, args.pool, resolve_sets(args.sets), (i, n), args.device,
        args.finalize_only, args.wavlm, args.hifigan)


if __name__ == "__main__":
    main()
