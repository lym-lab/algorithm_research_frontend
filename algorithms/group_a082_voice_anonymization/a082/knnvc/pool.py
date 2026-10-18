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
from collections import OrderedDict
from pathlib import Path

import torch
from tqdm import tqdm

from .. import audio, seeding
from ..models import load_wavlm, wavlm_features

MAX_FRAMES = 30000   # 10 min at 50 frames/s
MIN_FRAMES = 15000   # speakers with less than 5 min after VAD are left out


def scan_voxceleb(root: str | Path) -> dict[str, list[str]]:
    """VoxCeleb1 layout: <root>/<id10270>/<video id>/<00001>.wav."""
    root = Path(root)
    pool = {}
    for spk_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("id")):
        files = sorted(str(f) for f in spk_dir.glob("*/*.wav"))
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
          wavlm_path=None) -> None:
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "frames.json").unlink(missing_ok=True)  # Pool rebuilds the index from the final cache
    wavlm = load_wavlm(device, wavlm_path)
    speakers = sorted(manifest)[shard[0]::shard[1]]
    for spk in tqdm(speakers, desc="matching sets"):
        out = cache / f"{spk}.pt"
        if out.exists():
            continue
        s = build_matching_set(wavlm, manifest[spk], spk)
        tmp = out.with_suffix(".tmp")
        torch.save(s, tmp)
        tmp.replace(out)


class Pool:
    """Cached matching sets with a small in-memory LRU (each set is about 60 MB in float16)."""

    def __init__(self, cache: str | Path, device="cuda", min_frames: int = MIN_FRAMES, keep: int = 8):
        self.cache = Path(cache)
        self.device = device
        self.keep = keep
        self._lru: OrderedDict[str, torch.Tensor] = OrderedDict()
        index = self.cache / "frames.json"
        if index.exists():
            frames = json.loads(index.read_text())
        else:
            frames = {p.stem: int(torch.load(p, map_location="cpu").shape[0])
                      for p in sorted(self.cache.glob("*.pt"))}
            index.write_text(json.dumps(frames, indent=0))
        self.speakers = sorted(s for s, n in frames.items() if n >= min_frames)
        if not self.speakers:
            raise RuntimeError(f"no speaker in {self.cache} has >= {min_frames} frames")

    def __len__(self) -> int:
        return len(self.speakers)

    def get(self, spk: str) -> torch.Tensor:
        if spk in self._lru:
            self._lru.move_to_end(spk)
            return self._lru[spk]
        s = torch.load(self.cache / f"{spk}.pt", map_location="cpu").to(self.device).float()
        self._lru[spk] = s
        if len(self._lru) > self.keep:
            self._lru.popitem(last=False)
        return s


def _parse_shard(s: str) -> tuple[int, int]:
    i, n = s.split("/")
    return int(i), int(n)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("scan-voxceleb"); a.add_argument("root"); a.add_argument("manifest")
    a = sub.add_parser("scan-tree"); a.add_argument("root"); a.add_argument("manifest"); a.add_argument("--ext", default=".wav")
    a = sub.add_parser("build"); a.add_argument("manifest"); a.add_argument("cache")
    a.add_argument("--device", default="cuda"); a.add_argument("--shard", default="0/1"); a.add_argument("--wavlm")
    args = ap.parse_args(argv)
    if args.cmd == "scan-voxceleb":
        m = scan_voxceleb(args.root)
    elif args.cmd == "scan-tree":
        m = scan_tree(args.root, args.ext)
    else:
        build(json.loads(Path(args.manifest).read_text()), args.cache, args.device,
              _parse_shard(args.shard), args.wavlm)
        return
    Path(args.manifest).write_text(json.dumps(m, indent=1))
    print(f"{len(m)} speakers, {sum(map(len, m.values()))} files -> {args.manifest}")


if __name__ == "__main__":
    main()
