import argparse
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from mutagen.mp3 import MP3
from torch.utils.data import Dataset
from .audio import read_audio
from .model import SAMPLE_RATE, SEGMENT_SAMPLES


def manifest_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_fma(root, output, pilot=False, seed=2026):
    root, output = Path(root), Path(output)
    if not pilot and not (root.parent / "fma.verified.json").exists():
        raise RuntimeError("Full FMA archive must be verified before paper split")
    rng = np.random.default_rng(seed)
    tracks = sorted(root.rglob("*.mp3"))
    rng.shuffle(tracks)
    records = {"train": [], "validation": []}
    targets = {"train": 8 * 24 if pilot else int(np.ceil(80 * 3600 * SAMPLE_RATE / (24 * SEGMENT_SAMPLES))) * 24,
               "validation": 2 * 24 if pilot else int(np.ceil(10 * 3600 * SAMPLE_RATE / (24 * SEGMENT_SAMPLES))) * 24}
    rejected = []
    for path in tracks:
        try:
            metadata = MP3(path).info
            # MP3 encoders may report 191999 bps for nominal 192 kbps.
            if metadata.bitrate < 191500:
                rejected.append([str(path), "bitrate<192kbps"])
                continue
            audio = read_audio(path)
            span = 24 * SEGMENT_SAMPLES
            if len(audio) < span:
                rejected.append([str(path), "too_short"])
                continue
            split = "train" if len(records["train"]) < targets["train"] else "validation"
            if len(records[split]) >= targets[split]:
                break
            start = int(rng.integers(0, len(audio) - span + 1))
            for i in range(24):
                records[split].append({"path": str(path.resolve()), "track_id": path.stem,
                                       "start": start + i * SEGMENT_SAMPLES, "length": SEGMENT_SAMPLES})
        except (RuntimeError, ValueError, OSError) as exc:
            rejected.append([str(path), str(exc)])
    if any(len(records[k]) < targets[k] for k in records):
        raise RuntimeError(f"Insufficient eligible data: {[len(v) for v in records.values()]} / {targets}")
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in records.items():
        (output / f"{split}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"pilot": pilot, "seed": seed, "rejected": rejected,
               "segments": {k: len(v) for k, v in records.items()},
               "hours": {k: len(v) * SEGMENT_SAMPLES / SAMPLE_RATE / 3600 for k, v in records.items()}}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def prepare_rwc(root, output, seed=2026):
    files = sorted(Path(root).rglob("*.wav"))
    groups = {}
    for path in files:
        match = re.fullmatch(r"RWC_G(\d{3})([ABC]?)", path.stem)
        if not match:
            raise ValueError(f"Unexpected RWC filename: {path.name}")
        groups.setdefault(match[1], []).append(path)
    if set(groups) != {f'{i:03d}' for i in range(1, 101)}:
        raise ValueError("RWC Genre must contain piece IDs 001 through 100")
    rng = np.random.default_rng(seed)
    rows = []
    for piece, parts in sorted(groups.items()):
        counts = [len(read_audio(path)) - SEGMENT_SAMPLES + 1 for path in parts]
        if min(counts) <= 0:
            raise ValueError(f"RWC {piece}: part shorter than one segment")
        boundaries = np.cumsum([0, *counts])
        starts = rng.choice(int(boundaries[-1]), 30, replace=False)
        for start in starts:
            part = int(np.searchsorted(boundaries[1:], start, side='right'))
            rows.append({"path": str(parts[part].resolve()), "track_id": f'RWC_G{piece}',
                         "start": int(start-boundaries[part]), "length": SEGMENT_SAMPLES})
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


class AudioDataset(Dataset):
    def __init__(self, manifest, mask_dir=None):
        self.rows = json.loads(Path(manifest).read_text(encoding="utf-8"))
        self.mask_dir = Path(mask_dir) if mask_dir else None
        # Cache a few decoded tracks rather than all float WAV/STFT data.
        self.read = lru_cache(maxsize=16)(read_audio)
        self.cache = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        audio = self.read(row["path"])[row["start"]:row["start"] + row["length"]].copy()
        result = {"audio": torch.from_numpy(audio), "index": index}
        if self.mask_dir:
            # One contiguous mmap avoids hundreds of thousands of small files.
            if self.cache is None:
                self.cache = np.load(self.mask_dir / "masks.npy", mmap_mode="r")
            result["mask"] = torch.from_numpy(np.array(self.cache[index]))
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["fma", "rwc"])
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--pilot", action="store_true")
    args = p.parse_args()
    if args.kind == "fma":
        print(prepare_fma(args.root, args.output, args.pilot))
    else:
        prepare_rwc(args.root, args.output)

if __name__ == "__main__":
    main()
