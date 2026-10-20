"""Target-speaker pools for kNN-VC.

A pool is a set of speakers, each with a matching set: WavLM layer-6 features of that speaker's
VAD-trimmed audio, at most `max_frames` frames (30000 = 10 min). Matching sets are computed once and
cached as float16 tensors, `<cache>/<spk>.pt`, so all systems and all data sets share them.

    python -m a082.knnvc.pool scan-voxceleb  <vox1 wav root> <manifest.json>
    python -m a082.knnvc.pool scan-tree      <corpus root>   <manifest.json>  [--ext .wav]
    python -m a082.knnvc.pool build <manifest.json> <cache dir> [--shard i/n] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path

import torch
from tqdm import tqdm

from .. import audio, seeding
from ..models import load_wavlm, wavlm_features

MAX_FRAMES = 30000   # 10 min at 50 frames/s
MIN_FRAMES = 15000   # speakers with less than 5 min after VAD are left out


def scan_voxceleb(root: str | Path, ext: str = ".wav") -> dict[str, list[str]]:
    """VoxCeleb layout: <root>/<id10270>/<video id>/<00001>.wav (VoxCeleb2: .m4a)."""
    root = Path(root)
    pool = {}
    for spk_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("id")):
        files = sorted(str(f) for f in spk_dir.glob(f"*/*{ext}"))
        if files:
            pool[spk_dir.name] = files
    return pool


def scan_tree(root: str | Path, ext: str = ".wav") -> dict[str, list[str]]:
    """LibriSpeech / LibriTTS-R layout: <root>/<speaker>/<chapter>/<file><ext>."""
    root = Path(root)
    pool = {}
    for spk_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(str(f) for f in spk_dir.glob(f"*/*{ext}"))
        if files:
            pool[spk_dir.name] = files
    return pool


@torch.inference_mode()
def build_matching_set(wavlm, files: list[str], spk: str, max_frames: int = MAX_FRAMES,
                       vad_trigger: float = 7.0) -> torch.Tensor:
    """Features of the speaker's files in a fixed shuffled order until `max_frames` are collected."""
    order = seeding.rng(spk, "pool-order").permutation(len(files))
    feats, n = [], 0
    for i in order:
        wav = audio.load_audio(files[i])
        wav = audio.vad_trim(wav, trigger_level=vad_trigger)
        if wav.shape[-1] < 16000 * 0.2:  # nothing left after VAD
            continue
        f = wavlm_features(wavlm, wav).half().cpu()
        feats.append(f)
        n += f.shape[0]
        if n >= max_frames:
            break
    if not feats:
        return torch.empty(0, 1024, dtype=torch.float16)
    return torch.cat(feats)[:max_frames]


def build(manifest: dict[str, list[str]], cache: str | Path, device="cuda", shard=(0, 1),
          wavlm_path=None, max_frames: int = MAX_FRAMES) -> None:
    """Write <cache>/<spk>.pt and its frame count <cache>/<spk>.frames. Each file is written once,
    atomically, by the shard that owns the speaker, so shards never touch a shared file."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    wavlm = load_wavlm(device, wavlm_path)
    speakers = sorted(manifest)[shard[0]::shard[1]]
    for spk in tqdm(speakers, desc="matching sets"):
        out, count = cache / f"{spk}.pt", cache / f"{spk}.frames"
        if out.exists() and count.exists():
            continue
        s = build_matching_set(wavlm, manifest[spk], spk, max_frames=max_frames)
        tmp = out.with_suffix(".tmp")
        torch.save(s, tmp)
        tmp.replace(out)
        count.with_suffix(".ftmp").write_text(str(s.shape[0]))
        count.with_suffix(".ftmp").replace(count)


class Pool:
    """Cached matching sets. Two LRU tiers: float16 copies in host memory (`keep_cpu`, ~60 MB each),
    float32 on the device (`keep`, ~120 MB each). The host tier matters for 1M, where consecutive
    utterances rarely share their second target: without it nearly every utterance reads a set from disk."""

    def __init__(self, cache: str | Path, device="cuda", min_frames: int = MIN_FRAMES, keep: int = 8,
                 keep_cpu: int | None = None, max_frames: int | None = None):
        # max_frames: use only the first frames of each stored set. Sets are built in a fixed per-speaker order,
        # so the first 30000 frames of an uncapped set are exactly the 10-minute set (one cache serves both)
        self.max_frames = max_frames
        # ~60 MB per set: 256 sets made each 1M process ~26 GB and 9 of them overran the machine's memory
        # (2026-10-20); a miss only costs a read from disk. A082_POOL_CPU_SETS overrides.
        keep_cpu = keep_cpu if keep_cpu is not None else int(os.environ.get("A082_POOL_CPU_SETS", "48"))
        self.cache = Path(cache)
        self.device, self.keep, self.keep_cpu = device, keep, keep_cpu
        # uncapped sets reach 1.5 GB in float32 (VoxCeleb1, up to ~2 h per speaker): with 3 processes per GPU, 8 of
        # them would overrun a 32 GB card, so the device tier is also bounded in bytes (10-minute sets never hit it)
        self.dev_bytes = int(float(os.environ.get("A082_POOL_GPU_MB", "3000")) * 2**20)
        self._dev: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._cpu: OrderedDict[str, torch.Tensor] = OrderedDict()
        pts = sorted(self.cache.glob("*.pt"))
        missing = [p.stem for p in pts if not p.with_suffix(".frames").exists()]
        if missing:
            raise RuntimeError(f"{len(missing)} matching sets in {self.cache} have no .frames file "
                               f"(build still running or interrupted?), e.g. {missing[:3]}")
        frames = {p.stem: int(p.with_suffix(".frames").read_text()) for p in pts}
        self.speakers = sorted(s for s, n in frames.items() if n >= min_frames)
        if not self.speakers:
            raise RuntimeError(f"no speaker in {self.cache} has >= {min_frames} frames")

    def __len__(self) -> int:
        return len(self.speakers)

    @staticmethod
    def _touch(lru: OrderedDict, key, value, limit, max_bytes: int | None = None):
        lru[key] = value
        lru.move_to_end(key)
        size = lambda: sum(t.numel() * t.element_size() for t in lru.values())
        while len(lru) > limit or (max_bytes is not None and len(lru) > 1 and size() > max_bytes):
            lru.popitem(last=False)

    def get(self, spk: str) -> torch.Tensor:
        if spk in self._dev:
            self._dev.move_to_end(spk)
            return self._dev[spk]
        host = self._cpu.get(spk)
        if host is None:
            host = torch.load(self.cache / f"{spk}.pt", map_location="cpu").half()
            if self.max_frames:
                host = host[: self.max_frames].clone()
        self._touch(self._cpu, spk, host, self.keep_cpu)
        s = host.to(self.device).float()
        self._touch(self._dev, spk, s, self.keep, self.dev_bytes)
        return s


def _parse_shard(s: str) -> tuple[int, int]:
    i, n = s.split("/")
    return int(i), int(n)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("scan-voxceleb"); a.add_argument("root"); a.add_argument("manifest"); a.add_argument("--ext", default=".wav")
    a = sub.add_parser("scan-tree"); a.add_argument("root"); a.add_argument("manifest"); a.add_argument("--ext", default=".wav")
    a = sub.add_parser("build"); a.add_argument("manifest"); a.add_argument("cache")
    a.add_argument("--device", default="cuda"); a.add_argument("--shard", default="0/1"); a.add_argument("--wavlm")
    a.add_argument("--max-frames", type=int, default=MAX_FRAMES, help="frames per speaker (50 per s); 0 = no limit")
    args = ap.parse_args(argv)
    if args.cmd == "scan-voxceleb":
        m = scan_voxceleb(args.root, args.ext)
    elif args.cmd == "scan-tree":
        m = scan_tree(args.root, args.ext)
    else:
        build(json.loads(Path(args.manifest).read_text()), args.cache, args.device,
              _parse_shard(args.shard), args.wavlm, args.max_frames or 10**9)
        return
    Path(args.manifest).write_text(json.dumps(m, indent=1))
    print(f"{len(m)} speakers, {sum(map(len, m.values()))} files -> {args.manifest}")


if __name__ == "__main__":
    main()
